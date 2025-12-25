from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # model_id = "Qwen/Qwen3-4B-Instruct-2507"
    model_id = "results/grpo/Qwen3-4B-Instruct-2507-GRPO-14B/checkpoint-1600-merged"

    sampling_params = SamplingParams(max_tokens=1024)
    llm = LLM(model_id, max_model_len=1024)
    PROMPT = "You are a careful rewrite assistant.\nRewrite the <TEXT> in {lang} so that every word, except proper nouns or proper adjectives, is at or below the {level} vocabulary level.\nReplace or simplify any other words above {level} level with easier alternatives while preserving the original meaning and coherence.\nDo not skip, shorten, or omit any part of the text. Keep sentence count and structure.\nOutput only the fully converted text with no explanations, instructions, or extra words.\n\n<TEXT>\n{shortened_text}"
    lang = "Korean"
    # shortened_text='Philosophy (from old Greek words meaning "love of wisdom") is the careful study of very basic questions about life, reason, knowledge, value, mind, and language. It is a way of thinking that looks closely at its own methods and ideas.\nIn the past, many sciences, like physics and psychology, were part of philosophy. Today, they are seen as different areas of study. Important traditions in the history of philosophy include Western, Arabic–Persian, Indian, and Chinese ways of thought. Western philosophy started in Ancient Greece and has many smaller areas. A key topic in Arabic–Persian philosophy is the link between reason and faith. Indian philosophy joins the problem of how to reach spiritual freedom with the study of reality and knowledge. Chinese philosophy often looks at practical matters like right social behavior, government, and personal growth.\nMain areas of philosophy are knowledge study, ethics, logic, and study of reality. The study of knowledge asks what knowledge is and how we can get it. Ethics studies rules of good and bad and what right action is. Logic is the study of correct thinking and shows how strong arguments are different from weak ones. The study of reality looks at the most general parts of the world, being, objects, and their qualities. Other parts are beauty, language, mind, religion, science, math, history, and politics. In each area, there are different groups that support other ideas, rules, or ways.\nPhilosophers use many ways to reach knowledge. These include looking at ideas, using common sense, testing thoughts in the mind, looking at everyday language, telling about experience, and asking deep questions. Philosophy is linked to many other fields, like science, math, business, law, and news writing. It gives a wide view and studies the main ideas of these fields. It also looks at their methods and moral effects.'
    shortened_text = "철학(Philosophy)이라는 용어는 고대 그리스어의 필로소피아(φιλοσοφία, 지혜에 대한 사랑)에서 유래하였는데, 여기서 지혜는 일상생활에서의 실용하는 지식이 아닌 인간 자신과 그것을 둘러싼 세계를 관조하는 지식을 뜻한다. 이를테면 세계관, 인생관, 가치관이 포함된다. 이런 일반적인 의미로서의 철학은 어느 문화권에나 오래전부터 존재하여 왔다. 고대 그리스에서는 사실 학문 그 자체를 논하는 단어였고 전통상으로는 세계와 인간과 사물과 현상의 가치와 궁극적인 뜻을 향한 본질적이고 총체적인 천착을 뜻했다. 동양의 서구화 이후 철학은 대체로 고대 그리스 철학에서 시작하는 서양철학 일반을 지칭하기도 하나 철학 자체는 동서로 분리되지 않는다. 이에 더하여 현대 철학은 철학에 기초한 사고인 전제나 문제 명확화, 개념 엄밀화, 명제 간 관계 명료화를 이용해 제 주제를 논하는 언어철학이나 논리학등에 상당한 비중을 두고 있다."
    # shortened_text = "歴史上、物理学や心理学など、多くの個別的な科学が哲学の一部から発生した。しかし、現代的には、これらの科学は哲学とは異なる分野として扱われている。歴史上、影響力のある哲学の伝統としては、西洋哲学、アラブ・ペルシア哲学、インド哲学、中国哲学などがある。西洋哲学は古代ギリシアに起源を持ち、哲学における幅広い下位分野をカバーする。アラブ・ペルシア哲学における主要なトピックは理性と啓示の関係であり、インド哲学はどのようにして悟りに達するかの精神的な問題と、現実の本質や知識にたどり着く方法の探求を結びつける。中国哲学は主に正しい社会的行動や統治、そして自己修養（英語版）に関する実践的な問題に焦点を置く。"
    # shortened_text = "哲学是研究普遍的、基本问题的学科，包括存在、知识、价值、理智、心灵、语言、人生、道德等领域。哲学与其他学科不同之处在於哲学有独特之思考方式，例如批判的方式、通常是系统化的方法，并以理性论证为基础。从历史上看，许多单独的学科，例如物理学、生物学等自然科学，或法学、政治学、心理学等社会科学，都曾被视作哲学的一部分或其分支学科。直至其得到后续发展后，才逐渐被视作现代意义上的独立学科。"
    LEVEL = "TOPIK Level 2"
    PROMPT = PROMPT.format(lang=lang, level=LEVEL, shortened_text=shortened_text)
    messages = [{"role": "user", "content": PROMPT}]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Setting enable_thinking=False disables thinking mode

    )

    outputs = llm.generate(text, sampling_params)
    print(outputs)
    print(outputs[0].outputs[0].text)
    # print(assessor.reward_vocab_level([outputs[0].outputs[0].text], ['TOPIK Level 2'] ,['ko']))

if __name__ == "__main__":
    main()

# only cefr_level
# Philosophy (from old Greek words meaning "love of wise things") is also school for also good also also also also also also

# 3000
# Philosophy (from old Greek words, which mean "love of wisdom") is the small house for learning about life, music, work, family, and music. This house is for children who want to know about the world. 
#
# In the beginning, some people learned about music and games in the garden of philosophy. After he died, he began to write about the music of England and Africa. He helped others to understand the music of Europe in 1800.
#
# In 1900, the music of France became famous in England. The British school of music began in 1910, and became important in 1920. Peter Smith liked the music of India in 1930, and he drew three pictures of the Indian house.

# 1000
# Philosophy (from old Greek words meaning "love of wisdom") is the small group that helps people with simple questions about life, mind, family, work, and school. This group does not have many books, but every person who likes philosophy writes small books or tells stories. In the beginning, people did not write much, but after many years, some wrote about the school, and others helped with music or dance.

# 500
# Philosophy (from old Greek words meaning "love of wisdom") is the careful study of simple questions about life, mind, money, school, music, and house. It is a way of writing and thinking about the same school and old books. In the year 1000, many people began to write about music and school. Some wrote in Greece, and some wrote in Japan, China, India, and America. After many years, some schools began to teach music from these books.

# re-500
# Philosophy (from small blue words showing “small number of new things”) is the small number of new games for very small new games about small red number, small name, new number, small red number, small name, and small blue number. It is a small number of new things that shows small new numbers and small red numbers.

# re3-200-quadratic-penalize
# Big big book (from old old big words and small small words) is the small small play with big big words and small small games. Big big book is the small small play and small small games.

# re4-bleu-penalty
# Math（from old British words meaning “like of big things”） is the very big work of very small questions about the world， using， money， work， black， mind， and words． This is a big way of thinking which looks very much at its big work and its ideas．

# no-entailment-9600
# Music. (From an old Greek word meaning “love of world.”) This is the slow study of very small important problems about life, life, life, life, body, and language. This is a different way of thinking which often studies its own orders and ideas.
# 한 과학. (Philosophy). 이 이름은 아주 오래 그리스어의 필로소피아(φιλοσοφία, 큰 문제에 대한 사랑)에서 시작된 것이다. 이곳에서 큰 문제는 큰 밀生活的 큰 학습이 아닌 사람 자신과 그것을 단 둘러싼 세계를 공부하는 학습을 뜻한다. 이를 통해서 세계관, 삶의 문제, 큰 문제를 포함한다. 이런 작은 큰 의미의 과학은 어느 문화권에나 오래 전부터 시작하여 왔다. 오래 전 그리스에서는 큰 학문 그 본체를 공부하는 이름이었다 그리고 옛 시간에 이전 세계와 사람과 큰 것과 큰 문제의 큰 문제와 가장 큰 큰 문제를 바라보는 큰 큰 중요한 큰 큰 밝은 학업을 원한 것이다. 동양의 서유화 이후 과학은 많은 때 오래 전 그리스 과학에서 시작하는 서유과학 큰 것을 바꾸기도 하나 과학 본체는 동서에서 큰 문제가 바뀌지 않는다. 이에 더하여 오늘의 과학은 과학 위에 마주하는 큰 생각인 큰 문제나 큰 문제를 더 큰 일이 되는 큰 문제, 큰 문제를 더 큰 문제로 바꾸는 큰 문제를 사용하여 큰 문제를 공부하는 언어과학이나 큰 과학등에 많이 큰 공을 주고 있다.
# この世界. より大きな science または ミュート学など、多くの小さな大きな science が大きな世界の一つから生まれた. しかし、今になって、これらの science は大きな世界と違う大きな世界として使われている. 歴史上、大きな大きな学校の古い大きな学問としては、ヨーロッパの学問、アラブ・ペルシア学問、インド学問、中国学問などがある.ヨーロッパの学問は古いギリシアに大きな問題を持っている、大きな世界の中で大きな小さな小さな大きな世界を大きな大きな世界にする. アラブ・ペルシア学問での大きな大きな問題は大きなものは大きな答えの大きな問題である、インド学問はどのようにして大きな答えに大きな問題を学ぶの大きな大きな問題と、大きな世界の大きな問題または大きな答えに大きな問題を学ぶ方法をよく知っている. 中国学問は大きな大きな大きな人生の問題または大きな学校の life、そして自分の life（英語版）についての大きな大きな問題に大きな大きな問題をする.
# Music. This is a study of often-known, small problems. This includes life, money, life problems, body life, head life, languages, life, life problems. Music is different from other studies. This is because music uses different type of study answers. For example, the small study type, often using systemized answers, and using small world answers. From the early life, many small different studies, for example physics, biology and other world sciences, or music studies, football studies, music studies and other school sciences, have often been thought to be one part of music or one of its small study studies. Only after this got more early study after, can often be thought to be an early-world small study.

# grpo-7400 entailment 0.5 bertscore 0.5
# We can say “philosophy” – we find it from old Greek words – we can find the meaning “love of wisdom” – we can find in the group – – we can say – – philosophy – – – from the word group – – – – we can find it – – – “philosophy” – – we think about – life, and also about reason or knowledge, we can find in the sentence – – we can say yes – so we can say it’s a nice group – we can find in the English sentence – yes, we can say it’s true.
# 철학이라는 단어를 말할 때, 우리는 "필로소피아"라고 하기도 하면서, 이것은 고대 그리스어에서 비롯된다고 생각해요 – 우리는 그거를 찾아보면, 지혜를 이야기하는 부분에서 나오는 것이라고 생각해요 – 이 문장에서 말하는 ‘지혜’는, 우리 평상시에 쓰는 지식이 아니라, 인간 자신을 보고, 그리고 그 세계를 보는 것을 말하는 걸로 생각해요 – 우리는 그걸 발견할 수 있죠. 그렇게 말하면, 예를 들어서 “세계관”이라는 것, 그리고 “인생관”, 그리고 “가치관”을 살펴볼 수 있는 예를 주면, 우리는 철학 안에 포함되는 게 된다고 생각할 수 있어요 – 그래서 우리는 그 예를 보면 아주 잘 설명할 수 있다고 생각해요, 그래서 그 문장에서 우리는 ‘철학’을 살펴보면 좋을 것 같아요 – 그래서 우리는 그 것을 찾을 수 있죠 – 우리는 그게 좋다 라고 생각해요.
# たった過去の歴史の中で、物理のことがり、心理のことがり、など、たくさんある科学たち（例えば物理学や心理学）が、哲学というグループの部分から生まれたと、私たちは言えると思っています – これは真実なんだ。

# grpo-qwen1.7b
# The, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in, a, thing, in,

# grpo-qwen-2400
# Called – – –, – –, – –, – –, – –, – –, – –.