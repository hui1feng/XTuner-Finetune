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



LoRA微调不需要太大显存开销，旁路分支模型Adapter文件也就是LoRA模型文件。
QLoRA是LoRA的改进
全参数微调、QLoRA、LoRA比较


全参数微调:模型和优化器全部加载到显存
LoRA:模型和LoRA部分的参数优化器加载到显存
QLoRA:4-bit方式简单加载模型，QLoRA部分的参数优化器可以在GPU和CPU之间调度，不怕显存爆。
2.XTuner


上手


自定义微调


训练完成得到LoRA文件


支持工具类模型对话


数据处理引擎


不需要处理复杂数据格式


多数据样本拼接，增强并行性





3.8GB显存玩转LLM


Flash Attention 加速训练、DeepSpeed ZeRO优化节省显存。
优化前后显存占用情况


4.动手实战

