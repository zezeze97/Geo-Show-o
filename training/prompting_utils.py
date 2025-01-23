# coding=utf-8
# Copyright 2024 NUS Show Lab.
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
                 special_tokens=("<|soi|>", "<|eoi|>", "<|t2i|>", "<|mmu|>", "<|mix|>"),
                 max_text_len=8000, ignore_id=-100):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]',
                                                'bos_token': '<|sot|>',
                                                'eos_token': '<|eot|>'})
        self.text_tokenizer.add_tokens(list(special_tokens))
        self.sptids_dict = {token: torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token])) for token in
                            special_tokens}
        
        self.sptids_dict['<|sot|>'] = torch.tensor([self.text_tokenizer.bos_token_id])
        self.sptids_dict['<|eot|>'] = torch.tensor([self.text_tokenizer.eos_token_id])
        self.sptids_dict['<|pad|>'] = torch.tensor([self.text_tokenizer.pad_token_id])
        # plus 1 because at this time we add a task token before
        self.max_text_len = max_text_len + 1
        self.pad_id = self.text_tokenizer.convert_tokens_to_ids('[PAD]')
        self.ignore_id = ignore_id
    
        
    def t2i_prompt(self, text_ids, image_ids, labels):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]


            if self.max_text_len >= len(temp_ids):
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * (len(temp_ids) + image_ids.shape[-1] + 2)
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                
            else:
                # should add the eos token
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 2)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat([
                # should we predict text tokens when doing image reconstruction?
                # torch.tensor(temp_ids).to(device),
                torch.ones_like(torch.tensor(temp_ids)).to(device) * self.ignore_id,
                self.sptids_dict['<|soi|>'].to(device),
                labels[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)

            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)
        
    
    def t2i_gen_prompt(self, text_ids):
        sequence_ids = []
        attention_masks = []
        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]

            if self.max_text_len >= len(temp_ids):
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * (len(temp_ids)  + 1)
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
            else:
                # should add the eos token
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids)  + 1)  # +1 for one special tokens
                
            temp_ids = torch.cat([
                torch.tensor(temp_ids),
                self.sptids_dict['<|soi|>'],
            ], dim=0)
            temp_masks = torch.tensor(temp_masks)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    def mmu_prompt(self, image_ids, instruction_ids, response_ids):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        max_text_len = self.max_text_len - 1
        assert image_ids.shape[0] == len(instruction_ids) == len(response_ids)
        for i in range(len(instruction_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []

            # add bos
            if len(instruction_ids[i]) == 0:
                instruction_ids[i] = [self.text_tokenizer.bos_token_id]
            elif instruction_ids[i][0] != self.text_tokenizer.bos_token_id:
                instruction_ids[i] = [self.text_tokenizer.bos_token_id] + instruction_ids[i]

            temp_ids = instruction_ids[i] + response_ids[i] + [self.text_tokenizer.eos_token_id]
            temp_label_ids =  [self.ignore_id] * len(instruction_ids[i]) + response_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                # minus 1 because task token was prepended to the former image tokens
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3) + [0] * (max_text_len - len(temp_ids))
                temp_ids = temp_ids + [self.pad_id] * (max_text_len - len(temp_ids))
                temp_label_ids = temp_label_ids + [self.ignore_id] * (max_text_len - len(temp_label_ids))
                
            else:
                # should add the eos token
                temp_ids = temp_ids[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_label_ids = temp_label_ids[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3)  # +3 for three special tokens
            # prompting -- [task token] [soi] [image tokens] [eoi] [sot] [text tokens] [eot] 
            temp_ids = torch.cat([
                self.sptids_dict['<|mmu|>'].to(device),  # task token
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                torch.tensor(temp_ids).to(device),
            ], dim=0)
            
            temp_label_ids = torch.cat([
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.ones_like(image_ids[i]) * self.ignore_id,
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor(temp_label_ids).to(device),
            ], dim=0)

            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)

            

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)
           
    def mix_prompt(self, text_ids, image_ids, labels):
        pass
        

    def __call__(self, input, task):
        """
        input (tuple) : data pairs contain text(str), image(tensor), or videos(tensor).
        task (str) : a flag indicates the current task.
        """
        if task == "t2i":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(text_ids, image_ids, input[2])
            

        elif task == "t2i_gen":
            text_ids = self.text_tokenizer(input)['input_ids']  # (B, max_len)
            sequence_ids_with_masks = self.t2i_gen_prompt(text_ids)
        
        elif task == "mmu":
            image_ids = input[0]
            instruction_ids = self.text_tokenizer(input[1])['input_ids']
            response_ids = self.text_tokenizer(input[2])['input_ids']
            sequence_ids_with_masks = self.mmu_prompt(image_ids, instruction_ids, response_ids)
        else:
            raise NotImplementedError

        return sequence_ids_with_masks

if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=40,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|t2i|>", "<|mmu|>", "<|mix|>"),
                                       ignore_id=-100)
    
    instructions = ['一只小猪',
                  '生成随机']
    responses = ['小黑猫',
                 '']
    image_input = torch.randint(0, 10, (2, 10))
    input, attention_mask = uni_prompting(instructions, task='t2i_gen')
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
    
    
    input, attention_mask, label = uni_prompting((image_input, instructions, responses), task='mmu')
    print('*'*10 + ' mmu ' + '*'*10)
    print(f'input: {input}')
    print(f'input shape: {input.shape}')
    print(f'attention_mask: {attention_mask}')
    print(f'attention_mask shape: {attention_mask.shape}')
    print(f'label: {label}')
    print(f'label shape: {label.shape}')
    