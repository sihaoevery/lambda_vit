import torch
import numpy
import os

layer = [0,1,2,3,4,5,6,7,8,9,10,11]

for i in layer:
    tag=f'{i}'

    root = '/root/checkpoint_saving'
    test_model = 'xx.pth'

    model_source =os.path.join(root,test_model)
    model_save = os.path.join(root,f'transplant_at_layer_{tag}.pth') 

    model_full = os.path.join(root,'deit_base_patch16_224-b5f2ef4d.pth')

    try:
        m_s = torch.load(model_source,'cpu')['model']
    except:
        m_s = torch.load(model_source,'cpu')['net']

    m_f = torch.load(model_full,'cpu')['model']

    # for i in layer:
    #     print(f'processing layer {i}')
    #     key=f'blocks.{i}.'
    #     for k in m_s.keys():
    #         if key in k:
    #             print(k)
    #             m_s[k]=m_f[k]

    print(f'processing layer {i}')
    key=f'blocks.{i}.'
    # key=f'head'
    for k in m_s.keys():
        if key in k:
            print(k)
            m_s[k]=m_f[k]
    a=dict()
    a['model']=m_s
    torch.save(a,model_save)
