# XTuner-Finetune
1.Finetune简介
![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/5d801ae6-9a87-4ae3-9619-4bb8a187e55f)

大语言模型在海量文本内容上，基于无监督和半监督进行训练的。
在具体场景中表现不尽如人意故需要微调。
微调模式：增量预训练和指令微调。
增量预训练：给某些投喂一些某研究领域的新知识。
指令跟随：预训练模型仅仅简单拟合训练集中的分布，为使模型更加服从指令，需要进行指令微调，得到instructed LLM。

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/9f854615-6890-4c57-83b2-c77f58273f43)

1.1微调具体实现方式：

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/f0b1011d-b0fa-4190-af72-9e395dad6317)

首先对训练数据进行角色指定：问题给User，答案给指定角色，完成对话模板构建。
开源模型使用的对话模板不同

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/a73611b7-dd88-44e5-bdc3-d1c8bb158327)

注意：在测试阶段，用户使用模型对话时，不需要角色分配。

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/bbfb3cee-d29e-427a-80e7-b3c64157c314)

训练时只会对答案计算Loss

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/050cb42e-8d35-4c76-8573-d59649e686e9)

增量微调不需要问题，是陈述句。

1.2.Xtuner微调原理

![image](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/302b9fa6-6590-4f82-a0ef-8f1577027c42)


LoRA微调不需要太大显存开销，旁路分支模型Adapter文件也就是LoRA模型文件。
QLoRA是LoRA的改进
全参数微调、QLoRA、LoRA比较

![imagee](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/61dbe0a8-e3bd-4276-a37f-2a39234410a1)


全参数微调:模型和优化器全部加载到显存
LoRA:模型和LoRA部分的参数优化器加载到显存
QLoRA:4-bit方式简单加载模型，QLoRA部分的参数优化器可以在GPU和CPU之间调度，不怕显存爆。
2.XTuner

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


3.8GB显存玩转LLM

![image7](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/935dcd6a-9885-40a2-9b69-b90444d40f96)


Flash Attention 加速训练、DeepSpeed ZeRO优化节省显存。
优化前后显存占用情况

![image8](https://github.com/hui1feng/XTuner-Finetune/assets/126125104/296116d8-7c30-46c9-899c-92e383bd0d3e)


4.动手实战

代码参考链接：https://github.com/InternLM/tutorial/tree/main/xtuner
视频学习资料：https://www.bilibili.com/video/BV1yK4y1B75J/?spm_id_from=333.788
