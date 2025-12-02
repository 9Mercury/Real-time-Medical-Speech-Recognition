import soundfile as sf

from dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B")

text = "[S1] Yuvraj Singh is a Majesty"

output = model.generate(text)

sf.write("simple.mp3", output, 44100)