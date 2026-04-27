# text embeding with clip
import os
import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

MOD = {'MR': 'magnetic resonance imaging', 'CT': 'computed tomography imaging'}
CLS = ['background','spleen', 'right kidney', 'left kidney', 'gall bladder',
        'esophagus', 'liver', 'stomach', 'arota', 'postcava', 'pancreas',
        'right adrenal gland', 'left adrenal gland', 'duodenum'] 

txt_encoding = []
with torch.no_grad():
    for mod in MOD:
        print(f'A {MOD[mod]}.')
        
        ## CLIP V3
        # text_inputs = torch.cat([clip.tokenize(f'A {MOD[mod]} of a {item}.') for item in CLS]).to(device)
        ## CLIP V1
        # text_inputs = torch.cat([clip.tokenize(f'A photo of a {item}.') for item in CLS]).to(device)
        ## CLIP V2
        text_inputs = torch.cat([clip.tokenize(f'There is a {item} in this {MOD[mod]}.') for item in CLS]).to(device)
        
        text_features = model.encode_text(text_inputs)
        print(text_features.shape, text_features.dtype)
        txt_encoding.append(text_features)

mod_cls_txt_encoding = torch.stack(txt_encoding)
print(mod_cls_txt_encoding.shape)
torch.save(mod_cls_txt_encoding, 'mod_cls_txt_encoding.pth')