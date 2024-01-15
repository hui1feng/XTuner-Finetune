# XTuner-Finetune
## 1.Finetune简介
![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/5d801ae6-9a87-4ae3-9619-4bb8a187e55f)

大语言模型在海量文本内容上，基于无监督和半监督进行训练的。
在具体场景中表现不尽如人意故需要微调。
微调模式：增量预训练和指令微调。
增量预训练：给某些投喂一些某研究领域的新知识。
指令跟随：预训练模型仅仅简单拟合训练集中的分布，为使模型更加服从指令，需要进行指令微调，得到instructed LLM。

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/9f854615-6890-4c57-83b2-c77f58273f43)

### 1.1微调具体实现方式：

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/f0b1011d-b0fa-4190-af72-9e395dad6317)

首先对训练数据进行角色指定：问题给User，答案给指定角色，完成对话模板构建。
开源模型使用的对话模板不同

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/a73611b7-dd88-44e5-bdc3-d1c8bb158327)

注意：在测试阶段，用户使用模型对话时，不需要角色分配。

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/bbfb3cee-d29e-427a-80e7-b3c64157c314)

训练时只会对答案计算Loss

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/050cb42e-8d35-4c76-8573-d59649e686e9)

增量微调不需要问题，是陈述句。

### 1.2.Xtuner微调原理

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/302b9fa6-6590-4f82-a0ef-8f1577027c42)


LoRA微调不需要太大显存开销，旁路分支模型Adapter文件也就是LoRA模型文件。
QLoRA是LoRA的改进
全参数微调、QLoRA、LoRA比较

![imagee](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/61dbe0a8-e3bd-4276-a37f-2a39234410a1)


全参数微调:模型和优化器全部加载到显存
LoRA:模型和LoRA部分的参数优化器加载到显存
QLoRA:4-bit方式简单加载模型，QLoRA部分的参数优化器可以在GPU和CPU之间调度，不怕显存爆。
##2.XTuner

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/27d435e9-718a-43f4-b49e-80b2fab9e5fb)

上手
![下载](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/e01852cd-d03c-4972-844e-f87a19c85df3)


自定义微调

![image0](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/f719b5e7-aab2-4222-9570-3bc8cb7fef50)

训练完成得到LoRA文件

![image1](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/b506ae43-1d95-4fd2-98b5-43740405c7c3)


支持工具类模型对话

![image2](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/286e98ef-2b03-4230-a0b5-6c244b64191c)

数据处理引擎

![image3](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/371b7a06-ce89-4684-81db-4160d2fb0832)


不需要处理复杂数据格式

![image4](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/7ed3f73d-9b6d-4fba-ba1e-b668f49bf401)


多数据样本拼接，增强并行性

![image5](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/60d701a1-90c3-41e1-935a-1e7cb6c6c209)


![image6](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/f5185178-4d45-40e5-88b3-4eae19e9943a)


## 3.8GB显存玩转LLM

![image7](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/935dcd6a-9885-40a2-9b69-b90444d40f96)


Flash Attention 加速训练、DeepSpeed ZeRO优化节省显存。
优化前后显存占用情况

![image8](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/296116d8-7c30-46c9-899c-92e383bd0d3e)


## 4.动手实战
### 4.1安装
```python
# 如果你是在 InternStudio 平台，则从本地 clone 一个已有 pytorch 2.0.1 的环境：
/root/share/install_conda_env_internlm_base.sh xtuner0.1.9
# 如果你是在其他平台：
conda create --name xtuner0.1.9 python=3.10 -y

# 激活环境
conda activate xtuner0.1.9
# 进入家目录 （~的意思是 “当前用户的home路径”）
cd ~
# 创建版本文件夹并进入，以跟随本教程
mkdir xtuner019 && cd xtuner019


# 拉取 0.1.9 的版本源码
git clone -b v0.1.9  https://github.com/InternLM/xtuner
# 无法访问github的用户请从 gitee 拉取:
# git clone -b v0.1.9 https://gitee.com/Internlm/xtuner

# 进入源码目录
cd xtuner

# 从源码安装 XTuner
pip install -e '.[all]'

# 创建一个微调 oasst1 数据集的工作路径，进入
mkdir ~/ft-oasst1 && cd ~/ft-oasst1
```
### 4.2微调
#### 4.2.1准备配置文件->模型下载->数据集下载
```python
cd ~/ft-oasst1
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
#由于下载模型很慢，用教学平台的同学可以直接复制模型。
cp -r /root/share/temp/model_repos/internlm-chat-7b ~/ft-oasst1/
#由于 huggingface 网络问题，咱们已经给大家提前下载好了，复制到正确位置即可：
cd ~/ft-oasst1
# ...-guanaco 后面有个空格和英文句号啊
cp -r /root/share/temp/datasets/openassistant-guanaco .
```
#### 4.2.2修改配置文件
```python
#修改其中的模型和数据集为 本地路径
cd ~/ft-oasst1
vim internlm_chat_7b_qlora_oasst1_e3_copy.py

# 修改internlm_chat_7b_qlora_oasst1_e3_copy.py中的模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm-chat-7b'

# 修改训练数据集为本地路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = './openassistant-guanaco'
```
#### 4.2.3开始微调
训练：

xtuner train ${CONFIG_NAME_OR_PATH}

也可以增加 deepspeed 进行训练加速：

xtuner train ${CONFIG_NAME_OR_PATH} --deepspeed deepspeed_zero2

例如，我们可以利用 QLoRA 算法在 oasst1 数据集上微调 InternLM-7B：

```python
# 单卡
## 用刚才改好的config文件训练
xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py

# 多卡
NPROC_PER_NODE=${GPU_NUM} xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py

# 若要开启 deepspeed 加速，增加 --deepspeed deepspeed_zero2 即可
```
 将得到的 PTH 模型转换为 HuggingFace 模型，即：生成 Adapter 文件夹
```python
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1

xtuner convert pth_to_hf ./internlm_chat_7b_qlora_oasst1_e3_copy.py ./work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_1.pth ./hf
```
此时，hf 文件夹即为我们平时所理解的所谓 “LoRA 模型文件”,可以简单理解：LoRA 模型文件 = Adapter.
### 4.3部署与测试
将 HuggingFace adapter 合并到大语言模型：
```python
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
# xtuner convert merge \
#     ${NAME_OR_PATH_TO_LLM} \
#     ${NAME_OR_PATH_TO_ADAPTER} \
#     ${SAVE_PATH} \
#     --max-shard-size 2GB
```

与合并后的模型对话：
```python
# 加载 Adapter 模型对话（Float 16）
xtuner chat ./merged --prompt-template internlm_chat

# 4 bit 量化加载
# xtuner chat ./merged --bits 4 --prompt-template internlm_chat
```
Demo
* 修改 cli_demo.py 中的模型路径
```python
- model_name_or_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b"
+ model_name_or_path = "merged"
```
* 运行 cli_demo.py 以目测微调效果
```python
python ./cli_demo.py
```
### 4.4 自定义微调
场景需求:基于 InternLM-chat-7B 模型，用 MedQA 数据集进行微调，将其往医学问答领域对齐。
#### 4.4.1 数据准备
* 将数据转为 XTuner 的数据格式
原格式：(.xlsx)->目标格式：(.jsonL)
```python
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```
* 执行 python 脚本，获得格式化后的数据集
  ```python
  python xlsx2jsonl.py
  ```
* 划分训练集和测试集
   ```python
  my .jsonL file looks like:
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
Step1, read the .jsonL file.
Step2, count the amount of the "conversation" elements.
Step3, randomly split all "conversation" elements by 7:3. Targeted structure is same as the input.
Step4, save the 7/10 part as train.jsonl. save the 3/10 part as test.jsonl
  ```
#### 4.4.2 开始自定义微调
```python
mkdir ~/ft-medqa && cd ~/ft-medqa
cp -r ~/ft-oasst1/internlm-chat-7b .
git clone https://github.com/InternLM/tutorial
cp ~/tutorial/xtuner/MedQA2019-structured-train.jsonl .
#准备配置文件
# 复制配置文件到当前目录
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
# 改个文件名
mv internlm_chat_7b_qlora_oasst1_e3_copy.py internlm_chat_7b_qlora_medqa2019_e3.py

# 修改配置文件内容
vim internlm_chat_7b_qlora_medqa2019_e3.py
#减号代表要删除的行，加号代表要增加的行。
# 修改import部分
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory

# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm-chat-7b'

# 修改训练数据为 MedQA2019-structured-train.jsonl 路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = 'MedQA2019-structured-train.jsonl'

# 修改 train_dataset 对象
train_dataset = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=data_path),
+   dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
-   dataset_map_fn=alpaca_map_fn,
+   dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)
#微调启动
xtuner train internlm_chat_7b_qlora_medqa2019_e3.py --deepspeed deepspeed_zero2
#pth 转 huggingface
#部署与测试
```
### 4.5【补充】用 MS-Agent 数据集 赋予 LLM 以 Agent 能力
概述
MSAgent 数据集每条样本包含一个对话列表（conversations），其里面包含了 system、user、assistant 三种字段。其中：

* system: 表示给模型前置的人设输入，其中有告诉模型如何调用插件以及生成请求

* user: 表示用户的输入 prompt，分为两种，通用生成的prompt和调用插件需求的 prompt

* assistant: 为模型的回复。其中会包括插件调用代码和执行代码，调用代码是要 LLM 生成的，而执行代码是调用服务来生成结果的

一条调用网页搜索插件查询“上海明天天气”的数据样本示例
#### 4.5.1微调步骤
```python
# 准备工作
mkdir ~/ft-msagent && cd ~/ft-msagent
cp -r ~/ft-oasst1/internlm-chat-7b .

# 查看配置文件
xtuner list-cfg | grep msagent

# 复制配置文件到当前目录
xtuner copy-cfg internlm_7b_qlora_msagent_react_e3_gpu8 .

# 修改配置文件中的模型为本地路径
vim ./internlm_7b_qlora_msagent_react_e3_gpu8_copy.py
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm-chat-7b'
```
由于 msagent 的训练非常费时，大家如果想尽快把这个教程跟完，可以直接从 modelScope 拉取咱们已经微调好了的 Adapter。如下演示：
开始 chat 之前，还要加个 serper 的环境变量：去 serper.dev 免费注册一个账号，生成自己的 api key。
```python
#下载Adapter
cd ~/ft-msagent
apt install git git-lfs
git lfs install
git lfs clone https://www.modelscope.cn/xtuner/internlm-7b-qlora-msagent-react.git
#添加 serper 环境变量
export SERPER_API_KEY=abcdefg
#xtuner + agent，启动！
xtuner chat ./internlm-chat-7b --adapter internlm-7b-qlora-msagent-react --lagent
```

代码参考链接：https://github.com/InternLM/tutorial/tree/main/xtuner
视频学习资料：https://www.bilibili.com/video/BV1yK4y1B75J/?spm_id_from=333.788
