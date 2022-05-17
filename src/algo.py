import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from src.arch import TeacherNet, StudentNet 
from src.sparse import Masking, CosineDecay
from src.datasets import GeneralSet, ClientSet
from torch.utils.data import DataLoader as ImageLoader
from sklearn.metrics import roc_auc_score as aucscore
from src.metrics import EER, BPCER10, HTER, OPT_HTER


#==============================================================================
#  Codes for Printing Logs, Writing Logs, and Saving Models.
#==============================================================================
def print_logs(opt, model_id, total_loss, test_auc=None, test_pf=None):
    """
    print train logs
    """
    print('==================================================================')
    print('[Model ID: %d][loss: %.5f]'%(model_id, total_loss))
    print('[Test AUC:%.4f]' % (test_auc))

    if test_pf != None:
        for l in range(len(test_pf)):
            print('[Threshold:%.4f][FAR:%.4f][FRR:%.4f][HTER:%.4f]' % (test_pf[l][3], test_pf[l][0], test_pf[l][1], test_pf[l][2]))    
    print('==================================================================')

def write_logs(opt, model_id, total_loss, test_auc=None, test_pf=None):
    """
    write train logs
    """
    path = 'logs/{}/{}/{}_{}/seed_{}/'.format(opt.tail, opt.source_set, opt.target_set, opt.mask, opt.seed)
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + 'logs.txt', 'a') as f:
        f.write('\n==================================================================')
        f.write('\n[Model ID: %d][loss: %.5f]'%(model_id, total_loss))
        f.write('\n[Test AUC:%.4f]' % (test_auc))
        if test_pf != None:
            for l in range(len(test_pf)):
                f.write('\n[Threshold:%.4f][FAR:%.4f][FRR:%.4f][HTER:%.4f]' % (test_pf[l][3], test_pf[l][0], test_pf[l][1], test_pf[l][2]))          
        f.write('\n==================================================================')

def save_models(opt=None, teacher=None, model_id=None):
    """ 
    Save models
    """
    
    path = 'models/{}/{}/{}_{}/seed_{}/{}/'.format(opt.tail, opt.source_set, opt.target_set, opt.mask, opt.seed, model_id)
    if not os.path.exists(path):
        os.makedirs(path)
    
    if teacher != None:
        torch.save(teacher.state_dict(), path + 'teacher.pt'.format(model_id))

def save_sparse_models(opt=None, student=None, model_id=None):
    """ 
    Save model
    Returns:
    None
    """
    
    path = 'models/{}/{}/{}_{}/seed_{}/{}/'.format(opt.tail, opt.source_set, opt.target_set, opt.mask, opt.seed, model_id)
    if not os.path.exists(path):
        os.makedirs(path)
    
    if student != None:
        torch.save(dense_to_sparse(student.state_dict()), path + 'student.pt'.format(model_id))

#===============================================================================
#   Codes for Training Teacher Network
#===============================================================================

def compute_loss_teacher(out, label, criterion):
    out_0 = out[label==0]
    out_1 = out[label==1]
    out = torch.cat((out_0, out_1), 0)

    gt_0 = torch.zeros_like(out_0)
    gt_1 = torch.ones_like(out_1)
    gt = torch.cat((gt_0, gt_1), 0)
    
    loss = criterion(out, gt)
    return loss

def prediction_teacher(teacher, dataloader):
    """predict samples using model
    Args:
        model: Module
        dataloader: dataloader
    Returns:
        preds: cpu tensor
        labels: cpu tensor
    """
    dataloader = tqdm(dataloader)
    preds = torch.empty(0).to(torch.device('cuda'))
    labels = torch.empty(0).to(torch.device('cuda'))
    with torch.no_grad():
        for data in dataloader:
            img, label = data
            img, label = img.to(torch.device('cuda')), label.to(torch.device('cuda'))
            pred, _, _, _ = teacher(img)
            pred = pred.mean(1).mean(1).mean(1)
            label = torch.clamp(label, min=0, max=1)
            preds = torch.cat((preds, pred), 0)
            labels = torch.cat((labels, label), 0)
    preds = preds.to(torch.device('cpu'))
    labels = labels.to(torch.device('cpu'))
    
    return preds, labels

def test_teacher(teacher, dataloader):
    teacher.eval()
    preds, labels = prediction_teacher(teacher=teacher, dataloader=dataloader)
    test_auc = aucscore(labels, preds)
    
    far_testeer, frr_testeer, hter_testeer, threshold_testeer = EER(pred=preds, label=labels)
    far_testhter, frr_testhter, hter_testhter, threshold_testhter = OPT_HTER(pred=preds, label=labels)
    
    #=========================================================================================================================
    testeer_pf = [far_testeer, frr_testeer, hter_testeer, threshold_testeer]
    testhter_pf = [far_testhter, frr_testhter, hter_testhter, threshold_testhter]
    #==========================================================================================================================
    test_pf = [testeer_pf, testhter_pf]

    return test_auc, test_pf

def train_teacher(opt):
    train_dataset = GeneralSet(name=opt.source_set, sub='train', mode=opt.ptc, sps=opt.sps)
    test_dataset = GeneralSet(name=opt.source_set, sub='test', mode=opt.ptc, sps=opt.sps)

    train_loader = ImageLoader(train_dataset, batch_size=opt.bs, shuffle=True, num_workers=opt.nw)
    test_loader = ImageLoader(test_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.nw)    

    teacher = TeacherNet(ic=3).to(torch.device('cuda'))

    criterion = nn.BCELoss().to(torch.device('cuda'))
    optimizer = optim.Adam(teacher.parameters(), lr=opt.lr)

    iterition = 0
    model_id = 0
    total_loss = 0
    for epoch in range(opt.epoch):
        train_loader = tqdm(train_loader)
        for i, data in enumerate(train_loader):
            iterition = iterition + 1
            teacher.train()
            img, label = data
            img, label = img.to(torch.device('cuda')), label.to(torch.device('cuda'))
            out, _, _, _ = teacher(img)
            label = torch.clamp(label, min=0, max=1).float()
            batch_loss = compute_loss_teacher(out=out, label=label, criterion=criterion)
            total_loss += batch_loss.item()/len(train_loader)
            train_loader.set_description(
                (
                    f"model_id:{model_id};"
                    # f"batch loss:{batch_loss.item():.5f};"
                    f"total loss:{total_loss:.5f};"
                    )
            )
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if iterition == opt.period:
                test_auc, test_pf = test_teacher(teacher=teacher, dataloader=test_loader)
                
                print_logs(opt=opt, model_id=model_id, total_loss=total_loss, test_auc=test_auc, test_pf=test_pf)
                write_logs(opt=opt, model_id=model_id, total_loss=total_loss, test_auc=test_auc, test_pf=test_pf)
                save_models(opt=opt, teacher=teacher, model_id=model_id)

                teacher.train()
                model_id = model_id + 1

                if model_id == 21:
                    print('Finish Training !')
                    exit()

                total_loss = 0

                iterition = 0

#=====================================================================================================================
def compute_similarity(opt, tf1=None, tf2=None, tf3=None, sf1=None, sf2=None, sf3=None):
    tf1 = tf1.view(tf1.shape[0], -1)
    tf2 = tf2.view(tf2.shape[0], -1)
    tf3 = tf3.view(tf3.shape[0], -1)
  
    sf1 = sf1.view(sf1.shape[0], -1)
    sf2 = sf2.view(sf2.shape[0], -1)
    sf3 = sf3.view(sf3.shape[0], -1)

    # compute cosine similarities
    s1 = 1 -  F.cosine_similarity(tf1, sf1, dim=1)
    s2 = 1 - F.cosine_similarity(tf2, sf2, dim=1)
    s3 = 1 - F.cosine_similarity(tf3, sf3, dim=1)
    
    s = (s1 + s2 + s3) / 3
    

 
    return s

def compute_loss_student(opt, tf1=None, tf2=None, tf3=None, sf1=None, sf2=None, sf3=None):
    s = compute_similarity(opt, tf1, tf2, tf3, sf1, sf2, sf3)
    loss = s.mean()
    
    return loss

def prediction_student(opt, teacher, student, dataloader):
    dataloader = tqdm(dataloader)
    preds = torch.empty(0).to(torch.device('cuda'))
    labels = torch.empty(0).to(torch.device('cuda'))
    with torch.no_grad():
        for data in dataloader:
            img, label = data
            img, label = img.to(torch.device('cuda')), label.to(torch.device('cuda'))
            
            tf1, tf2, tf3 = teacher.embedding(img)
            tf1, tf2, tf3 = teacher.resize(tf1, tf2, tf3)
            sf1, sf2, sf3 = student.embedding(img)
            sf1, sf2, sf3 = student.resize(sf1, sf2, sf3)
            
            pred = compute_similarity(opt, tf1=tf1, tf2=tf2, tf3=tf3, sf1=sf1, sf2=sf2, sf3=sf3)
            preds = torch.cat((preds, pred), 0)

            label = torch.clamp(label, min=0, max=1)
            labels = torch.cat((labels, label), 0)
            
    preds = preds.to(torch.device('cpu'))
    labels = labels.to(torch.device('cpu'))
    
    return preds, labels

def test_student_ideal(opt, teacher, student, dataloader):
    teacher.eval()
    student.eval()
    preds, labels = prediction_student(opt=opt, teacher=teacher, student=student, dataloader=dataloader)
 
    test_auc = aucscore(labels, preds)
        
    far_testeer, frr_testeer, hter_testeer, threshold_testeer = EER(pred=preds, label=labels)
    far_testhter, frr_testhter, hter_testhter, threshold_testhter = OPT_HTER(pred=preds, label=labels)
        
    testeer_pf = [far_testeer, frr_testeer, hter_testeer, threshold_testeer]
    testhter_pf = [far_testhter, frr_testhter, hter_testhter, threshold_testhter]
    test_pf = [testeer_pf, testhter_pf]

    return test_auc, test_pf

def test_student_challenging(opt, teacher, student, val_dataloader, test_dataloader):
    teacher.eval()
    student.eval()
    
    preds, labels = prediction_student(opt=opt, teacher=teacher, student=student, dataloader=val_dataloader)
    _, val_threshold = BPCER10(pred=preds, label=labels)
    
    preds, labels = prediction_student(opt=opt, teacher=teacher, student=student, dataloader=test_dataloader)
    test_auc = aucscore(labels, preds)

    far_val_threshold, frr_val_threshold, hter_val_threshold = HTER(pred=preds, label=labels, threshold=val_threshold)

    val_threshold_pf = [far_val_threshold, frr_val_threshold, hter_val_threshold, val_threshold]
    
    test_pf = [val_threshold_pf]

    return test_auc, test_pf

def train_student(opt):
    if opt.target_set == 'client':
        train_dataset = ClientSet(sub='adaptation', client=opt.mask, sps=opt.sps)
        test_dataset = ClientSet(sub='test', client=opt.mask, sps=opt.sps)
        
        train_loader = ImageLoader(train_dataset, batch_size=opt.bs, shuffle=True, num_workers=opt.nw)
        test_loader = ImageLoader(test_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.nw)
    
    else:
        if opt.setting == 'ideal':
            train_dataset = GeneralSet(name=opt.target_set, sub='train', mode='adaptation', sps=opt.sps)
            test_dataset = GeneralSet(name=opt.target_set, sub='test', mode='grand', sps=opt.sps)
            
            train_loader = ImageLoader(train_dataset, batch_size=opt.bs, shuffle=True, num_workers=opt.nw)
            test_loader = ImageLoader(test_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.nw)

        if opt.setting == 'challenging':
            train_dataset = GeneralSet(name=opt.target_set, sub='train_x', mode='adaptation', sps=opt.sps)
            val_dataset = GeneralSet(name=opt.target_set, sub='val_x', mode='adaptation', sps=opt.sps)
            test_dataset = GeneralSet(name=opt.target_set, sub='test', mode='grand', sps=opt.sps)

            train_loader = ImageLoader(train_dataset, batch_size=opt.bs, shuffle=True, num_workers=opt.nw)
            val_loader = ImageLoader(val_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.nw)
            test_loader = ImageLoader(test_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.nw)
    
    teacher = TeacherNet(ic=3).to(torch.device('cuda'))
    teacher.load_state_dict(torch.load(opt.teacher_path))
    teacher.eval()
    
    student = StudentNet(ic=3).to(torch.device('cuda'))
    
    if opt.density == 0.1:
        prune_rate = 0.5
    
    if opt.density == 0.01:
        prune_rate = 0.2

    optimizer = optim.Adam(student.parameters(), lr=opt.lr)
    decay = CosineDecay(prune_rate=prune_rate, T_max=3000)
    
    masking = Masking(optimizer=optimizer, prune_rate_decay=decay, prune_rate=prune_rate)
    
    masking.add_module(module=student, density=opt.density)
    
    iteration = 0
    total_loss = 0
    model_id = 0
    
    for epoch in range(opt.epoch):
        for i, data in enumerate(train_loader):
            iteration = iteration + 1
            student.train()
            img, label = data
            img, label = img.to(torch.device('cuda')), label.to(torch.device('cuda'))
            
            tf1, tf2, tf3 = teacher.embedding(img)
            tf1, tf2, tf3 = teacher.resize(tf1, tf2, tf3)
            sf1, sf2, sf3 = student.embedding(img)
            sf1, sf2, sf3 = student.resize(sf1, sf2, sf3)

            label = torch.clamp(label, min=0, max=1).float()
            
            batch_loss = compute_loss_student(opt, tf1=tf1, tf2=tf2, tf3=tf3, sf1=sf1, sf2=sf2, sf3=sf3)
            total_loss += batch_loss.item()/opt.period
            
            optimizer.zero_grad()
            batch_loss.backward()
            masking.step()
        
            if iteration == opt.period:
                if opt.setting == 'ideal':
                    test_auc, test_pf= test_student_ideal(opt=opt, teacher=teacher, student=student, dataloader=test_loader)   
                
                if opt.setting == 'challenging':
                    test_auc, test_pf= test_student_challenging(
                        opt=opt, teacher=teacher, student=student, val_dataloader=val_loader, test_dataloader=test_loader)

                print_logs(opt=opt, model_id=model_id, total_loss=total_loss, test_auc=test_auc, test_pf=test_pf)
                write_logs(opt=opt, model_id=model_id, total_loss=total_loss, test_auc=test_auc, test_pf=test_pf)            
                save_sparse_models(opt=opt, student=student, model_id=model_id)
                
                masking.param_regrowth()
                iteration = 0
                total_loss = 0
                model_id = model_id + 1
                
                if model_id == 25:
                    print('Finish Training!')
                    exit()
   
                student.train()


#===================================================================================================================
def evaluation_student(opt):
    teacher = TeacherNet(ic=3).to(torch.device('cuda'))
    teacher.load_state_dict(torch.load(opt.teacher_path))
    teacher.eval()
    
    student = StudentNet(ic=3).to(torch.device('cuda'))
    student.load_state_dict(sparse_to_dense(torch.load(opt.student_path)))
    student.eval()

    if opt.target_set == 'client':
        test_dataset = ClientSet(sub='test', client=opt.mask, sps=opt.sps)
        test_loader = ImageLoader(test_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.nw)
    
    else:
        if opt.setting == 'ideal':
            test_dataset = GeneralSet(name=opt.target_set, sub='test', mode=opt.ptc, sps=opt.sps)
            test_loader = ImageLoader(test_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.nw)

        if opt.setting == 'challenging':
            val_dataset = GeneralSet(name=opt.target_set, sub='val_x', mode='adaptation', sps=opt.sps)
            test_dataset = GeneralSet(name=opt.target_set, sub='test', mode=opt.ptc, sps=opt.sps)
            
            val_loader = ImageLoader(val_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.nw)
            test_loader = ImageLoader(test_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.nw)

    if opt.setting == 'ideal':
        test_auc, test_pf = test_student_ideal(opt, teacher=teacher, student=student, dataloader=test_loader)
    else:
        test_auc, test_pf= test_student_challenging(
                        opt=opt, teacher=teacher, student=student, val_dataloader=val_loader, test_dataloader=test_loader)
    
    print_logs(opt=opt, model_id=0, total_loss=0, test_auc=test_auc, test_pf=test_pf)
    write_logs(opt=opt, model_id=0, total_loss=0, test_auc=test_auc, test_pf=test_pf)


#===================================================================================================================

def dense_to_sparse(stats):
    keys = [
        'conv1.0.weight',
        'Block1.0.weight',
        'Block1.3.weight',
        'Block1.6.weight',
        'Block2.0.weight',
        'Block2.3.weight',
        'Block2.6.weight',
        'Block3.0.weight',
        'Block3.3.weight',
        'Block3.6.weight'
        ]

    for key in keys:
        if key == 'conv1.0.weight':
            stats[key] = stats[key].view(64*3*3*3).to_sparse()
            #print(stats[key])

        if key == 'Block1.0.weight':
            stats[key] = stats[key].view(128*64*3*3).to_sparse()
            #print(stats[key])

        if key == 'Block1.3.weight':
            stats[key] = stats[key].view(196*128*3*3).to_sparse()
            #print(stats[key])

        if key == 'Block1.6.weight':
            stats[key] = stats[key].view(128*196*3*3).to_sparse()
            #print(stats[key])

        if key == 'Block2.0.weight':
            stats[key] = stats[key].view(128*128*3*3).to_sparse()
            #print(stats[key])

        if key == 'Block2.3.weight':
            stats[key] = stats[key].view(196*128*3*3).to_sparse()
            #print(stats[key])

        if key == 'Block2.6.weight':
            stats[key] = stats[key].view(128*196*3*3).to_sparse()
            #print(stats[key])

        if key == 'Block3.0.weight':
            stats[key] = stats[key].view(128*128*3*3).to_sparse()
            #print(stats[key]) 

        if key == 'Block3.3.weight':
            stats[key] = stats[key].view(196*128*3*3).to_sparse()
            #print(stats[key])

        if key == 'Block3.6.weight':
            stats[key] = stats[key].view(128*196*3*3).to_sparse()
            #print(stats[key])

    return stats

def sparse_to_dense(stats):
    keys = [
        'conv1.0.weight',
        'Block1.0.weight',
        'Block1.3.weight',
        'Block1.6.weight',
        'Block2.0.weight',
        'Block2.3.weight',
        'Block2.6.weight',
        'Block3.0.weight',
        'Block3.3.weight',
        'Block3.6.weight'
        ]

    for key in keys:
        if key == 'conv1.0.weight':
            stats[key] = stats[key].to_dense().view(64, 3, 3, 3)

        if key == 'Block1.0.weight':
            stats[key] = stats[key].to_dense().view(128, 64, 3, 3) 

        if key == 'Block1.3.weight':
            stats[key] = stats[key].to_dense().view(196, 128, 3, 3)

        if key == 'Block1.6.weight':
            stats[key] = stats[key].to_dense().view(128, 196, 3, 3)

        if key == 'Block2.0.weight':
            stats[key] = stats[key].to_dense().view(128, 128, 3, 3) 

        if key == 'Block2.3.weight':
            stats[key] = stats[key].to_dense().view(196, 128, 3, 3)

        if key == 'Block2.6.weight':
            stats[key] = stats[key].to_dense().view(128, 196, 3, 3)

        if key == 'Block3.0.weight':
            stats[key] = stats[key].to_dense().view(128, 128, 3, 3)  

        if key == 'Block3.3.weight':
            stats[key] = stats[key].to_dense().view(196, 128, 3, 3)

        if key == 'Block3.6.weight':
            stats[key] = stats[key].to_dense().view(128, 196, 3, 3)

    return stats
