from kokoro import KPipeline
# from IPython.display import display, Audio
import soundfile as sf
import torch
# pipeline = KPipeline(lang_code='a')


# text = '''
# [Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
# '''
# text = '''
# My name is mango. I am a teacher in pre-school. I love to teach children. I have a pet cat named kitty. She is very cute and playful. I also enjoy reading books and going for walks in the park.
# '''


# Chinese
pipeline = KPipeline(lang_code='z')

text = '''
我叫刘晓雪, 你好.
'''

generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    # display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000)
