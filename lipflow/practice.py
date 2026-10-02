"""Practice sentences and the clips you record for them, shared by the macOS and Windows setup.

Clips are saved with their known text under <Lipflow home>/clips/onboarding. Training holds
N_HELD_OUT of them out and only keeps the face model if it reads those better than the stock model.
"""
from __future__ import annotations

import glob
import os
import random
import re
import time

import numpy as np

from .paths import HOME as DIR

CLIPS = os.path.join(DIR, "clips", "onboarding")
N_SENTENCES = 24
N_HELD_OUT = 6

# Harvard sentences (IEEE 1969 "Recommended Practice for Speech Quality Measurements", lists 1-6):
# short, phonetically balanced sentences, so practice covers every lip shape evenly. Used when there's
# no Wispr Flow history to practice your own sentences from.
HARVARD = [
    "The birch canoe slid on the smooth planks", "Glue the sheet to the dark blue background",
    "It's easy to tell the depth of a well", "These days a chicken leg is a rare dish",
    "Rice is often served in round bowls", "The juice of lemons makes fine punch",
    "The box was thrown beside the parked truck", "The hogs were fed chopped corn and garbage",
    "Four hours of steady work faced us", "A large size in stockings is hard to sell",
    "The boy was there when the sun rose", "A rod is used to catch pink salmon",
    "The source of the huge river is the clear spring", "Kick the ball straight and follow through",
    "Help the woman get back to her feet", "A pot of tea helps to pass the evening",
    "Smoky fires lack flame and heat", "The soft cushion broke the man's fall",
    "The salt breeze came across from the sea", "The girl at the booth sold fifty bonds",
    "The small pup gnawed a hole in the sock", "The fish twisted and turned on the bent hook",
    "Press the pants and sew a button on the vest", "The swan dive was far short of perfect",
    "The beauty of the view stunned the young boy", "Two blue fish swam in the tank",
    "Her purse was full of useless trash", "The colt reared and threw the tall rider",
    "It snowed rained and hailed the same morning", "Read verse out loud for pleasure",
    "Hoist the load to your left shoulder", "Take the winding path to reach the lake",
    "Note closely the size of the gas tank", "Wipe the grease off his dirty face",
    "Mend the coat before you go out", "The wrist was badly strained and hung limp",
    "The stray cat gave birth to kittens", "The young girl gave no clear response",
    "The meal was cooked before the bell rang", "What joy there is in living",
    "A king ruled the state in the early days", "The ship was torn apart on the sharp reef",
    "Sickness kept him home the third week", "The wide road shimmered in the hot sun",
    "The lazy cow lay in the cool grass", "Lift the square stone over the fence",
    "The rope will bind the seven books at once", "Hop over the fence and plunge in",
    "The friendly gang left the drug store", "Mesh wire keeps chicks inside",
    "The frosty air passed through the coat", "The crooked maze failed to fool the mouse",
    "Adding fast leads to wrong sums", "The show was a flop from the very start",
    "A saw is a tool used for making boards", "The wagon moved on well oiled wheels",
    "March the soldiers past the next hill", "A cup of sugar makes sweet fudge",
    "Place a rosebush near the porch steps", "Both lost their lives in the raging storm",
]


CHINESE = [
    "今天下午我们一起讨论这个问题", "请把文件发给我的同事", "明天上午在办公室见面", "这个计划还需要继续完善",
    "我想先了解事情的经过", "大家可以提出自己的意见", "会议结束以后给你回复", "请不要修改原来的内容",
    "我们需要更多时间准备", "这个消息已经得到确认", "请帮我检查一下文字", "下次见面再详细介绍",
    "这件事情没有那么简单", "谢谢你提供这些信息", "我会尽快完成这项工作", "现在还不能确定结果",
    "请告诉我你的具体安排", "今天的天气非常不错", "我们可以换一种方法", "请把重要的内容记录下来",
    "这份报告需要重新整理", "我已经收到了你的邮件", "请等一会再开始会议", "我们明天继续这个话题",
    "请先确认大家都有时间", "这个地方距离公司很近", "我希望听到不同的建议", "请把问题说得更清楚一些",
    "我们应该认真考虑这件事", "这个项目正在顺利进行", "请在下班之前给我回复", "大家的努力取得了进展",
    "我会把结果告诉大家", "还有一些细节需要检查", "请不要忘记带上文件", "我们需要保持联系",
    "今天的工作已经完成", "明天还有重要的任务", "请仔细阅读这份材料", "这个方案可以继续改进",
    "我想听听你的看法", "我们一起解决这个问题", "请给我一个明确的答复", "这个决定需要大家同意",
    "我正在准备相关资料", "会议的时间还没有确定", "请把最新的消息告诉我", "我们可以再讨论一次",
    "这项工作需要共同完成", "请检查一下有没有错误", "我会按照计划开始工作", "这个结果令人满意",
    "请尽快安排下一次会议", "我们需要更加详细的信息", "今天先完成最重要的事情", "明天早上我会联系你",
    "请注意文件中的说明", "这个问题已经解决了", "我们还有很多事情要做", "谢谢大家的理解和支持",
]


def practice_sentences(n: int = N_SENTENCES, language="en") -> list[str]:
    """Half your own everyday sentences (from an imported Wispr Flow history: 5–12 words, no digits)
    for your real vocabulary, half Harvard sentences for even coverage of lip shapes; all Harvard
    if there's no history. Shuffled together."""
    from .personal import PHRASES
    if language == "zh":
        return random.sample(CHINESE, min(n, len(CHINESE)))
    mine = []
    if os.path.exists(PHRASES):
        for line in open(PHRASES, encoding="utf-8"):
            for s in re.split(r"(?<=[.!?])\s+", line.strip()):
                w = s.split()
                if 5 <= len(w) <= 12 and not re.search(r"\d|http|@|/", s):
                    mine.append(s.rstrip(".!?,"))
    random.shuffle(mine)
    own = list(dict.fromkeys(mine))[:n // 2]
    harvard = random.sample(HARVARD, n - len(own))
    out = own + harvard
    random.shuffle(out)
    return out


def saved_clips(language="en") -> list[dict]:
    items = []
    for p in sorted(glob.glob(os.path.join(CLIPS if language == "en" else os.path.join(CLIPS, language), "*.npz"))):
        d = np.load(p, allow_pickle=True)
        items.append({"rois": d["rois"], "text": str(d["text"]), "path": p})
    return items


def save_clip(rois, text: str, raw: str = "", language="en") -> str:
    folder = CLIPS if language == "en" else os.path.join(CLIPS, language)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{int(time.time() * 1000)}.npz")
    np.savez_compressed(path, rois=rois, text=text, raw=raw or "")
    return path
