# 开源数据和工具

## Awesome List

1. https://github.com/faroit/awesome-python-scientific-audio

2. https://github.com/wenet-e2e/speech-synthesis-paper

3. https://github.com/ddlBoJack/Speech-Resources

4. https://github.com/sindresorhus/awesome

## 参考书籍

1. [神经网络与深度学习](https://nndl.github.io/)

2. Tan X, Qin T, Soong F, et al. A survey on Neural Speech Synthesis[J]. arXiv preprint arXiv:2106.15561, 2021.

3.  Sisman B, Yamagishi J, King S, et al. An Overview of Voice Conversion and Its Challenges: From Statistical Modeling to Deep Learning[J]. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2020, 29: 132-157.

## 语音相关的会议、期刊、比赛和公司

### 会议

1. INTERSPEECH（Conference of the International Speech Communication Association）

2. ICASSP（IEEE International Conference on Acoustics, Speech and Signal Processing）

3. ASRU（IEEE Automatic Speech Recognition and Understanding Workshop）

4. ISCSLP（International Symposium on Chinese Spoken Language Processing）

5. ACL（Association of Computational Linguistics）

### 期刊

1. Computer Speech and Language

### 最新论文

1. [低调奋进TTS最新论文集](http://yqli.tech/page/tts_paper.html)

2. https://arxiv.org/list/eess.AS/recent

3. https://arxiv.org/list/cs.SD/recent

4. https://arxiv.org/list/cs.CL/recent

5. https://arxiv.org/list/cs.MM/recent

### 比赛

1. [Blizzard Challenge](http://www.festvox.org/blizzard/)

2. [Zero Resource Speech Challenge](https://www.zerospeech.com/)

3. [ICASSP2021 M2VoC](http://challenge.ai.iqiyi.com/detail?raceId=5fb2688224954e0b48431fe0)

4. [Voice Conversion Challenge](http://www.vc-challenge.org/)

5. CHiME: Computational Hearing in Multisource Environment

6. NIST

### 公司

1. [微软](https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/#features)

2. [谷歌云](https://cloud.google.com/text-to-speech/docs/voices?hl=zh-cn)

3. [捷通华声](https://www.aicloud.com/dev/ability/index.html?key=tts#ability-experience)

4. [Nuance](https://www.nuance.com/omni-channel-customer-engagement/voice-and-ivr/text-to-speech.html#!)

5. [Amazon polly](https://aws.amazon.com/cn/polly/)

6. [百度（翻译）](https://fanyi.baidu.com/)

7. [搜狗开发平台](https://ai.sogou.com/product/audio_composition/)

8. [搜狗（翻译）](https://fanyi.sogou.com/)

9. [有道开放平台](https://ai.youdao.com/product-tts.s)

10. [有道（翻译）](http://fanyi.youdao.com)

11. [微软（翻译）](https://cn.bing.com/translator)

12. [Google翻译](https://translate.google.cn/)

### 微信公众号

1. 阿里语音AI

2. CCF语音对话与听觉专委会

3. CSMT

4. 声学挖掘机

5. 谈谈语音技术

6. THUsatlab

7. WeNet步行街

8. 音频语音与语言处理研究组

9. 雨石记

10. 语音算法组

11. 语音杂谈

12. 语音之家

13. 智能语音新青年

## 开源资料

### 中文数据集

1. [标贝中文标准女声音库](https://www.data-baker.com/open_source.html):
    中文单说话人语音合成数据集，质量高。

2. [THCHS-30](https://www.openslr.org/18/):
    中文多说话人数据集，原为语音识别练手级别的数据集，也可用于多说话人中文语音合成。

3. [Free ST Chinese Mandarin Corpus](https://www.openslr.org/38/): 855个说话人，每个说话人120句话，有对应人工核对的文本，共102600句话。

4. [zhvoice](https://github.com/KuangDD/zhvoice): zhvoice语料由8个开源数据集，经过降噪和去除静音处理而成，说话人约3200个，音频约900小时，文本约113万条，共有约1300万字。

5. [滴滴800+小时DiDiSpeech语音数据集](https://arxiv.org/abs/2010.09275): DiDi开源数据集，800小时，48kHz，6000说话人，存在对应文本，背景噪音干净，适用于音色转换、多说话人语音合成和语音识别，参见：https://zhuanlan.zhihu.com/p/268425880。

6. [SpiCE-Corpus](https://github.com/khiajohnson/SpiCE-Corpus): SpiCE是粤语和英语会话双语语料库。

7. [HKUST](http://www.paper.edu.cn/scholar/showpdf/MUT2IN4INTD0Exwh):
    10小时，单说话人，采样率8kHz。

8. [AISHELL-1](https://www.aishelltech.com/kysjcp):
    170小时，400个说话人，采样率16kHz。

9. [AISHELL-2](http://www.aishelltech.com/aishell_2): 1000小时，1991个说话人，采样率44.1kHz。希尔贝壳开源了不少中文语音数据集，AISHELL-2是最近开源的一个1000小时的语音数据库，禁止商用。官网上还有其它领域，比如用于语音识别的4个开源数据集。

10.[AISHELL-3](https://www.aishelltech.com/aishell_3): 85小时，218个说话人，采样率44.1kHz。

### 英文数据集

1. [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): 英文单说话人语音合成数据集，质量较高，25小时，采样率22.05kHz。

2. [VCTK](https://datashare.is.ed.ac.uk/handle/10283/2651): 英文多说话人语音数据集，44小时，109个说话人，每人400句话，采样率48kHz，位深16bits。

3. [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1): 630个说话人，8个美式英语口音，每人10句话，采样率16kHz，位深16bits。[这里是具体下载地址](http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3)，下载方法：首先下载种子，然后执行：
    ```shell
    ctorrent *.torrent
    ```

4. [CMU ARCTIC](http://festvox.org/cmu_arctic/packed/): 7小时，7个说话人，采样率16kHz。语音质量较高，可以用于英文多说话人的训练。

5. [Blizzard-2011](https://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/): 16.6小时，单说话人，采样率16kHz。可以从[The Blizzard Challenge](https://www.cstr.ed.ac.uk/projects/blizzard/)查找该比赛的相关数据，从[SynSIG](https://www.synsig.org/index.php)查找该比赛的相关信息。

6. [Blizzard-2013](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/): 319小时，单说话人，采样率44.1kHz。

7. [LibriSpeech](https://www.openslr.org/12): 982小时，2484个说话人，采样率16kHz。[OpenSLR](https://www.openslr.org/resources.php)搜集了语音合成和识别常用的语料。

8. [LibriTTS](https://www.openslr.org/60): 586小时，2456个说话人，采样率24kHz。

9. [VCC 2018](https://datashare.ed.ac.uk/handle/10283/3061): 1小时，12个说话人，采样率22.05kHz。类似的，可以从[The Voice Conversion Challenge 2016](https://datashare.ed.ac.uk/handle/10283/2211)获取2016年的VC数据。

10. [HiFi-TTS](http://www.openslr.org/109/): 300小时，11个说话人，采样率44.1kHz。

11. [TED-LIUM](https://www.openslr.org/7/): 118小时，666个说话人。

12. [CALLHOME](https://catalog.ldc.upenn.edu/LDC97S42): 60小时，120个说话人，采样率8kHz。

13. [RyanSpeech](https://github.com/roholazandie/ryan-tts): 10小时，单说话人，采样率44.1kHz。交互式语音合成语料。

### 情感数据集

1. [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data): 用于语音合成和语音转换的情感数据集。

2. [情感数据和实验总结](https://github.com/Emotional-Text-to-Speech/dl-for-emo-tts): 实际是情感语音合成的实验总结，包含了一些情感数据集的总结。

### 其它数据集

1. [Opencpop](https://wenet.org.cn/opencpop): 高质量歌唱合成数据集。

2. [好未来开源数据集](https://ai.100tal.com/dataset): 目前主要开源了3个大的语音数据集，分别是语音识别数据集，语音情感数据集和中英文混合语音数据集，都是多说话人教师授课音频。

3. [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut): 日语，10小时，单说话人，采样率48kHz。

4. [KazakhTTS](https://github.com/IS2AI/Kazakh_TTS): 哈萨克语，93小时，2个说话人，采样率44.1/48kHz。

5. [Ruslan](https://ruslan-corpus.github.io/): 俄语，31小时，单说话人，采样率44.1kHz。

6. [HUI-Audio-Corpus](https://github.com/iisys-hof/HUI-Audio-Corpus-German): 德语，326小时，122个说话人，采样率44.1kHz。

7. [M-AILABS](https://github.com/imdatsolak/m-ailabs-dataset): 多语种，1000小时，采样率16kHz。

8. [India Corpus](https://data.statmt.org/pmindia/): 多语种，39小时，253个说话人，采样率48kHz。

9. [MLS](http://www.openslr.org/94/): 多语种，5.1万小时，6千个说话人，采样率16kHz。

10. [CommonVoice](https://commonvoice.mozilla.org/zh-CN/datasets): 多语种，2500小时，5万个说话人，采样率48kHz。

11. [CSS10](https://github.com/Kyubyong/css10): 十个语种的单说话人语音数据的集合，140小时，采样率22.05kHz。

12. [OpenSLR](https://www.openslr.org/resources.php): OpenSLR是一个专门托管语音和语言资源的网站，例如语音识别训练语料库和与语音识别相关的软件。迄今为止，已经有100+语音相关的语料。

13. [DataShare](https://datashare.ed.ac.uk/): 爱丁堡大学维护的数据集汇总，包含了语音、图像等多个领域的数据集和软件，语音数据集中包括了语音合成、增强、说话人识别、语音转换等方面的内容。

14. [Speech in Microsoft Research Open Data](https://msropendata.com/datasets?term=speech): 微软开源数据搜索引擎中关于语音的相关数据集。

15. [voice datasets](https://github.com/jim-schwoebel/voice_datasets): Github上较为全面的开源语音和音乐数据集列表，包括语音合成、语音识别、情感语音数据集、语音分离、歌唱等语料，找不到语料可以到这里看看。

16. [Open Speech Corpora](https://github.com/JRMeyer/open-speech-corpora): 开放式语音数据库列表，特点是包含多个语种的语料。

17. [EMIME](https://www.emime.org/participate.html): 包含一些TTS和ASR模型，以及一个中文/英语，法语/英语，德语/英语双语数据集。

18. [Celebrity Audio Extraction](https://github.com/celebrity-audio-collection/videoprocess): 中国名人数据集，包含中国名人语音和图像数据。

### 开源工具

1. [sonic](https://github.com/waywardgeek/sonic): 语音升降速工具。

2. [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz): 从语音识别工具Kaldi中提取出来的音素-音频对齐工具，可以利用MFA获取每一个音素的时长，供预标注或时长模型使用。

3. [宾西法尼亚大学强制对齐标注软件（P2FA）](https://github.com/jaekookang/p2fa_py3)：[这里](https://blog.csdn.net/jojozhangju/article/details/51951622)有相关的介绍，对于噪音数据鲁棒性差。

4. [ABXpy](https://github.com/bootphon/ABXpy): 语音等测评ABX测试网页。

5. [SpeechSubjectiveTest](https://github.com/bigpon/SpeechSubjectiveTest): 主观测评工具，包括用于语音合成和转换的MOS、PK（倾向性测听）、说话人相似度测试和ABX测试。

6. [Matools](https://github.com/matpool/matools): 机器学习环境配置工具库

7. [MyTinySTL](https://github.com/Alinshans/MyTinySTL): 基于C++11的迷你STL。

8. [CppPrimerPractice](https://github.com/applenob/Cpp_Primer_Practice): 《C++ Primer 中文版（第 5 版）》学习仓库。

9. [git-tips](https://github.com/521xueweihan/git-tips): Git的奇技淫巧。

### 开源项目

1. [coqui-ai TTS](https://github.com/coqui-ai/TTS): 采用最新研究成果构建的语音合成后端工具集。

2. [ESPNet](https://github.com/espnet/espnet): 语音合成和识别工具集，主要集成声学模型、声码器等后端模型。

3. [fairseq](https://github.com/pytorch/fairseq): 序列到序列建模工具，包含语音识别、合成、机器翻译等模型。

4. [eSpeak NG Text-to-Speech](https://github.com/espeak-ng/espeak-ng): 共振峰生成的语音合成模型，集成超过100个语种和口音的语音合成系统，特别地，可借鉴该项目中的多语种文本前端。

5. [Epitran](https://github.com/dmort27/epitran): 将文本转换为IPA的工具，支持众多语种。

6. [Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2): Tensorflow版本的Tacotron-2.

7. [Transformer TTS](https://github.com/as-ideas/TransformerTTS): TensorFlow 2实现的FastSpeech系列语音合成。

8. [Text-to-speech in (partially) C++ using Tacotron model + Tensorflow](https://github.com/syoyo/tacotron-tts-cpp): 采用TensorFlow C++ API运行Tacotron模型。

9. [muzic](https://github.com/microsoft/muzic): 微软AI音乐的开源项目，包括乐曲理解、音乐生成等多种工作。

10. [merlin](https://github.com/CSTR-Edinburgh/merlin): CSTR开发的统计参数语音合成工具包，需要与文本前端（比如Festival）和声码器（比如STRAIGHT或WORLD）搭配使用。