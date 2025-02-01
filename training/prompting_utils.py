# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
# TODO - SHOULD BE FURTHER IMPROVED.
class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 special_tokens=("<|sot|>", "<|eot|>", "<|soi|>", "<|eoi|>", "<|t2i|>", "<|formalization|>", "<|reasoning|>", "<|step|>", "<|conclusion|>"),
                 max_len=8000, ignore_id=-100):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.text_tokenizer.add_tokens(list(special_tokens))
        self.sptids_dict = {token: torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token])) for token in
                            special_tokens}
        
        self.sptids_dict['<｜begin▁of▁sentence｜>'] = torch.tensor([self.text_tokenizer.bos_token_id])
        self.sptids_dict['<｜end▁of▁sentence｜>'] = torch.tensor([self.text_tokenizer.eos_token_id])
        self.sptids_dict['<|pad|>'] = torch.tensor([self.text_tokenizer.pad_token_id])
        self.sptids_dict['<｜User｜>'] = torch.tensor([self.text_tokenizer.convert_tokens_to_ids('<｜User｜>')])
        self.sptids_dict['<｜Assistant｜>'] = torch.tensor([self.text_tokenizer.convert_tokens_to_ids('<｜Assistant｜>')])
        self.max_len = max_len
        self.pad_id = self.text_tokenizer.convert_tokens_to_ids('[PAD]')
        self.ignore_id = ignore_id
        self.assistant_id = self.text_tokenizer.convert_tokens_to_ids('<｜Assistant｜>')
    
        
    def t2i_prompt(self, text_ids, image_ids, labels):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        for i in range(len(text_ids)):
            # prompting -- <｜begin▁of▁sentence｜> [User] [task token] [sot] [text tokens] [eot] [Assistant] [soi] [image tokens] [eoi] <｜end▁of▁sentence｜>
            temp_text_ids =  [self.text_tokenizer.bos_token_id] + [int(self.sptids_dict['<｜User｜>'])] + [int(self.sptids_dict['<|t2i|>'])]  + \
                [int(self.sptids_dict['<|sot|>'])] + text_ids[i] + [int(self.sptids_dict['<|eot|>'])] + [int(self.sptids_dict['<｜Assistant｜>'])]
            temp_ids  = torch.cat([
                torch.tensor(temp_text_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)
            ], dim=0)
            
            temp_label_ids = torch.cat([
                # should we predict text tokens when doing image reconstruction?
                # torch.tensor(temp_text_ids).to(device),
                torch.ones_like(torch.tensor(temp_text_ids)).to(device) * self.ignore_id,
                self.sptids_dict['<|soi|>'].to(device),
                labels[i],
                self.sptids_dict['<|eoi|>'].to(device),
                self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)
            ], dim=0)  
        
            assert temp_ids.shape[0] == temp_label_ids.shape[0]
            
            current_len = temp_ids.shape[0]       
            if self.max_len >= current_len:
                temp_masks =  [1] * current_len + [0] * (self.max_len - current_len)
                temp_ids = torch.cat([temp_ids,
                                      torch.tensor([self.pad_id] * (self.max_len - current_len)).to(device)
                    ], dim=0)
                temp_label_ids = torch.cat([temp_label_ids,
                                            torch.tensor([self.ignore_id] * (self.max_len - current_len)).to(device)])
                
            else:
                # should add the eos token
                temp_ids = torch.cat([temp_ids[:self.max_len - 1],
                                      self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)],
                                     dim=0)
                temp_label_ids = torch.cat([temp_label_ids[:self.max_len-1],
                                            self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)])
                temp_masks = [1] * temp_ids.shape[0]
            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)
        
    
    def t2i_gen_prompt(self, text_ids):
        sequence_ids = []
        attention_masks = []
        assert len(text_ids) == 1
        for i in range(len(text_ids)):
            # prompting -- <｜begin▁of▁sentence｜> [User] [task token] [sot] [text tokens] [eot] [Assistant] [soi]
            temp_text_ids =  [self.text_tokenizer.bos_token_id] + [int(self.sptids_dict['<｜User｜>'])] + [int(self.sptids_dict['<|t2i|>'])] + \
                [int(self.sptids_dict['<|sot|>'])] + text_ids[i] + [int(self.sptids_dict['<|eot|>'])] + [int(self.sptids_dict['<｜Assistant｜>'])]
            temp_ids = torch.cat([
                torch.tensor(temp_text_ids),
                self.sptids_dict['<|soi|>'],
            ], dim=0)
            temp_masks = [1] * temp_ids.shape[0]
            temp_masks = torch.tensor(temp_masks)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    def mmu_prompt(self, task_token, image_ids, instruction_ids, response_ids):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        assert image_ids.shape[0] == len(instruction_ids) == len(response_ids)
        for i in range(len(instruction_ids)):
            # prompting -- <｜begin▁of▁sentence｜> [User] [task token] [soi] [image tokens] [eoi] [sot] [text tokens] [eot] <｜end▁of▁sentence｜> 
            response_ids[i] = [self.assistant_id] + response_ids[i]
            temp_text_ids = instruction_ids[i] + response_ids[i]
            temp_label_ids =  [self.ignore_id] * len(instruction_ids[i]) + response_ids[i]
            temp_ids = torch.cat([
                self.sptids_dict['<｜begin▁of▁sentence｜>'].to(device),
                self.sptids_dict['<｜User｜>'].to(device),
                self.sptids_dict[task_token].to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                self.sptids_dict['<|sot|>'].to(device),
                torch.tensor(temp_text_ids).to(device),
                self.sptids_dict['<|eot|>'].to(device),
                self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)
            ], dim=0)
            
            temp_label_ids = torch.cat([
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.ones_like(image_ids[i]) * self.ignore_id,
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor(temp_label_ids).to(device),
                self.sptids_dict['<|eot|>'].to(device),
                self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)
            ], dim=0)

            assert temp_ids.shape[0] == temp_label_ids.shape[0]
            
            current_len = temp_ids.shape[0]     
            
            if self.max_len >= current_len:
                temp_masks =  [1] * current_len + [0] * (self.max_len - current_len)
                temp_ids = torch.cat([temp_ids,
                                      torch.tensor([self.pad_id] * (self.max_len - current_len)).to(device)
                    ], dim=0)
                temp_label_ids = torch.cat([temp_label_ids,
                                            torch.tensor([self.ignore_id] * (self.max_len - current_len)).to(device)])
                
            else:
                # should add the eos token
                temp_ids = torch.cat([temp_ids[:self.max_len - 1],
                                      self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)],
                                     dim=0)
                temp_label_ids = torch.cat([temp_label_ids[:self.max_len-1],
                                            self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)])
                temp_masks = [1] * temp_ids.shape[0]
            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)
    
    def mmu_gen_prompt(self, task_token, image_ids, instruction_ids):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        assert len(instruction_ids) == 1
        for i in range(len(instruction_ids)):
            # prompting -- <｜begin▁of▁sentence｜> [User][task token] [soi] [image tokens] [eoi] [sot] [text tokens] [Assistant]
            temp_text_ids =  instruction_ids[i]
            temp_ids = torch.cat([
                self.sptids_dict['<｜begin▁of▁sentence｜>'].to(device),
                self.sptids_dict['<｜User｜>'].to(device),
                self.sptids_dict[task_token].to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                self.sptids_dict['<|sot|>'].to(device),
                torch.tensor(temp_text_ids).to(device),
                self.sptids_dict['<｜Assistant｜>'].to(device),
            ])
            temp_masks = [1] * temp_ids.shape[0]
            temp_masks = torch.tensor(temp_masks)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)
        
        
        
        
           
    def mix_prompt(self, text_ids, image_ids, labels):
        pass
        

    def __call__(self, input, task):
        """
        input (tuple) : data pairs contain text(str), image(tensor), or videos(tensor).
        task (str) : a flag indicates the current task.
        """
        if task == "t2i":
            text_ids = self.text_tokenizer(input[0], add_special_tokens=False)['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(text_ids, image_ids, input[2])
            

        elif task == "t2i_gen":
            text_ids = self.text_tokenizer([input], add_special_tokens=False)['input_ids']  # (B, max_len)
            sequence_ids_with_masks = self.t2i_gen_prompt(text_ids)
        
        elif task == "formalization":
            image_ids = input[0]
            instruction_ids = self.text_tokenizer(input[1], add_special_tokens=False)['input_ids']
            response_ids = self.text_tokenizer(input[2], add_special_tokens=False)['input_ids']
            sequence_ids_with_masks = self.mmu_prompt("<|formalization|>", image_ids, instruction_ids, response_ids)
            
        elif task == 'reasoning':
            image_ids = input[0]
            instruction_ids = self.text_tokenizer(input[1], add_special_tokens=False)['input_ids']
            response_ids = self.text_tokenizer(input[2], add_special_tokens=False)['input_ids']
            sequence_ids_with_masks = self.mmu_prompt("<|reasoning|>", image_ids, instruction_ids, response_ids)
        elif task == 'formalization_gen':
            image_ids = input[0]
            instruction_ids = self.text_tokenizer([input[1]], add_special_tokens=False)['input_ids']
            sequence_ids_with_masks = self.mmu_gen_prompt("<|formalization|>", image_ids, instruction_ids)
        elif task == 'reasoning_gen':
            image_ids = input[0]
            instruction_ids = self.text_tokenizer([input[1]], add_special_tokens=False)['input_ids']
            sequence_ids_with_masks = self.mmu_gen_prompt("<|reasoning|>", image_ids, instruction_ids)      
            
        else:
            raise NotImplementedError

        return sequence_ids_with_masks

if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    uni_prompting = UniversalPrompting(tokenizer, max_len=40,
                                       ignore_id=-100)
    
    instructions = ['一只小猪',
                  '生成随机']
    responses = ['小黑猫',
                 '']
    image_input = torch.randint(0, 10, (2, 10))
    input, attention_mask = uni_prompting(instructions[0], task='t2i_gen')
    print('*'*10 + ' t2i_gen ' + '*'*10)
    print(f'input: {input}')
    print(f'input shape: {input.shape}')
    print(f'attention_mask: {attention_mask}')
    print(f'attention_mask shape: {attention_mask.shape}')
    
    input, attention_mask, label = uni_prompting((instructions, image_input, image_input), task='t2i')
    print('*'*10 + ' t2i ' + '*'*10)
    print(f'input: {input}')
    print(f'input shape: {input.shape}')
    print(f'attention_mask: {attention_mask}')
    print(f'attention_mask shape: {attention_mask.shape}')
    print(f'label: {label}')
    print(f'label shape: {label.shape}')
    
    
    input, attention_mask, label = uni_prompting((image_input, instructions, responses), task='formalization')
    print('*'*10 + ' mmu ' + '*'*10)
    print(f'input: {input}')
    print(f'input shape: {input.shape}')
    print(f'attention_mask: {attention_mask}')
    print(f'attention_mask shape: {attention_mask.shape}')
    print(f'label: {label}')
    print(f'label shape: {label.shape}')
    
    input, attention_mask = uni_prompting([image_input[0, :].unsqueeze(0), instructions[0]], task='reasoning_gen')
    print('*'*10 + ' mmu_gen ' + '*'*10)
    print(f'input: {input}')
    print(f'input shape: {input.shape}')
    print(f'attention_mask: {attention_mask}')
    print(f'attention_mask shape: {attention_mask.shape}')
    