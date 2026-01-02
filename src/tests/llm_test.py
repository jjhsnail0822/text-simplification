from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # model_id = "Qwen/Qwen3-4B-Instruct-2507"
    model_id = "results/grpo/Qwen3-4B-Instruct-2507-2.5-1.0-2.0-jaccard/checkpoint-200-merged"

    sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)
    llm = LLM(model_id, max_model_len=1024)
    PROMPT = "You are a careful rewrite assistant.\nRewrite the <TEXT> in {lang} so that every word, except proper nouns or proper adjectives, is at or below the {level} vocabulary level.\nReplace or simplify any other words above {level} level with easier alternatives while preserving the original meaning and coherence.\nDo not skip, shorten, or omit any part of the text. Keep sentence count and structure.\nOutput only the fully converted text with no explanations, instructions, or extra words.\n\n<TEXT>\n{shortened_text}"
    lang = "English"
    shortened_text='Philosophy (from old Greek words meaning "love of wisdom") is the careful study of very basic questions about life, reason, knowledge, value, mind, and language. It is a way of thinking that looks closely at its own methods and ideas.\nIn the past, many sciences, like physics and psychology, were part of philosophy. Today, they are seen as different areas of study. Important traditions in the history of philosophy include Western, Arabic–Persian, Indian, and Chinese ways of thought. Western philosophy started in Ancient Greece and has many smaller areas. A key topic in Arabic–Persian philosophy is the link between reason and faith. Indian philosophy joins the problem of how to reach spiritual freedom with the study of reality and knowledge. Chinese philosophy often looks at practical matters like right social behavior, government, and personal growth.\nMain areas of philosophy are knowledge study, ethics, logic, and study of reality. The study of knowledge asks what knowledge is and how we can get it. Ethics studies rules of good and bad and what right action is. Logic is the study of correct thinking and shows how strong arguments are different from weak ones. The study of reality looks at the most general parts of the world, being, objects, and their qualities. Other parts are beauty, language, mind, religion, science, math, history, and politics. In each area, there are different groups that support other ideas, rules, or ways.\nPhilosophers use many ways to reach knowledge. These include looking at ideas, using common sense, testing thoughts in the mind, looking at everyday language, telling about experience, and asking deep questions. Philosophy is linked to many other fields, like science, math, business, law, and news writing. It gives a wide view and studies the main ideas of these fields. It also looks at their methods and moral effects.'
    # shortened_text = "철학(Philosophy)이라는 용어는 고대 그리스어의 필로소피아(φιλοσοφία, 지혜에 대한 사랑)에서 유래하였는데, 여기서 지혜는 일상생활에서의 실용하는 지식이 아닌 인간 자신과 그것을 둘러싼 세계를 관조하는 지식을 뜻한다. 이를테면 세계관, 인생관, 가치관이 포함된다. 이런 일반적인 의미로서의 철학은 어느 문화권에나 오래전부터 존재하여 왔다. 고대 그리스에서는 사실 학문 그 자체를 논하는 단어였고 전통상으로는 세계와 인간과 사물과 현상의 가치와 궁극적인 뜻을 향한 본질적이고 총체적인 천착을 뜻했다. 동양의 서구화 이후 철학은 대체로 고대 그리스 철학에서 시작하는 서양철학 일반을 지칭하기도 하나 철학 자체는 동서로 분리되지 않는다. 이에 더하여 현대 철학은 철학에 기초한 사고인 전제나 문제 명확화, 개념 엄밀화, 명제 간 관계 명료화를 이용해 제 주제를 논하는 언어철학이나 논리학등에 상당한 비중을 두고 있다."
    # shortened_text = "歴史上、物理学や心理学など、多くの個別的な科学が哲学の一部から発生した。しかし、現代的には、これらの科学は哲学とは異なる分野として扱われている。歴史上、影響力のある哲学の伝統としては、西洋哲学、アラブ・ペルシア哲学、インド哲学、中国哲学などがある。西洋哲学は古代ギリシアに起源を持ち、哲学における幅広い下位分野をカバーする。アラブ・ペルシア哲学における主要なトピックは理性と啓示の関係であり、インド哲学はどのようにして悟りに達するかの精神的な問題と、現実の本質や知識にたどり着く方法の探求を結びつける。中国哲学は主に正しい社会的行動や統治、そして自己修養（英語版）に関する実践的な問題に焦点を置く。"
    # shortened_text = "哲学是研究普遍的、基本问题的学科，包括存在、知识、价值、理智、心灵、语言、人生、道德等领域。哲学与其他学科不同之处在於哲学有独特之思考方式，例如批判的方式、通常是系统化的方法，并以理性论证为基础。从历史上看，许多单独的学科，例如物理学、生物学等自然科学，或法学、政治学、心理学等社会科学，都曾被视作哲学的一部分或其分支学科。直至其得到后续发展后，才逐渐被视作现代意义上的独立学科。"
    
    # shortened_text = "Pigeon photography is an aerial photography technique invented in 1907 by the German apothecary Julius Neubronner, who also used pigeons to deliver medications. A homing pigeon was fitted with an aluminium breast harness to which a lightweight time-delayed miniature camera could be attached. Neubronner's German patent application was initially rejected, but was granted in December 1908 after he produced authenticated photographs taken by his pigeons. He publicized the technique at the 1909 Dresden International Photographic Exhibition, and sold some images as postcards at the Frankfurt International Aviation Exhibition and at the 1910 and 1911 Paris Air Shows.\n\nInitially, the military potential of pigeon photography for aerial reconnaissance appeared interesting. Battlefield tests in World War I provided encouraging results, but the ancillary technology of mobile dovecotes for messenger pigeons had the greatest impact. Owing to the rapid development of aviation during the war, military interest in pigeon photography faded and Neubronner abandoned his experiments. The idea was briefly resurrected in the 1930s by a Swiss clockmaker, and reportedly also by the German and French militaries. Although war pigeons were deployed extensively during World War II, it is unclear to what extent, if any, birds were involved in aerial reconnaissance. The United States Central Intelligence Agency (CIA) later developed a battery-powered camera designed for espionage pigeon photography; details of its use remain classified."
    # shortened_text = "イギリス（イングランド、ウェールズ）では、評決が下されるまでの間、事件に関する報道を厳しく制限することにより、陪審への影響の防止を図っている。すなわち、制定法やコモン・ローにより、マスメディアの事件報道に対し、重い罰金（場合によっては拘禁）などの制裁を伴う強い規制を課している。審理前には、関係者の名前や予備審問の日時・場所のような最低限の情報しか報道してはならない。予備審問等は一般に公開されているものの、その内容を広く伝えることは規則によって禁じられている。審理が始まった後も、報道は手続を正確に伝えるものでなければならず、現在又は将来の手続（まだ審理が始まっていない別件の手続も含む）に害を及ぼすようなものであってはならない。これらに違反した場合は法廷侮辱罪による処罰の対象となり（実際上、処罰されるのは審理に深刻な影響を与える実質的な危険がある場合に限られている）、時々、法廷侮辱罪による処罰が行われる例がある。スコットランド、アイルランドも概ね同様の規制を敷いており、オーストラリア、ニュージーランド、カナダでは、これより緩やかな規制をしている。"
    # shortened_text = "양자 전기역학에 의하면 전자기적 상호작용은 두 입자가 광자를 교환하면서 발생한다. 주변에 아무것도 없이 혼자서 등속운동만 하는 전자가 에너지 보존 법칙이나 운동량 보존 법칙을 어겨가면서 광자를 흡수하거나 방출할 수는 없는 노릇이다. 그런데 주변에 전하를 띈 물체가 놓이게 되면, 가상 전자가 두 물체 사이에서 운동량을 서로 교환해주고, 이에 따라 쿨롱힘이 발생한다. 이처럼 쿨롱힘에 의해 궤적이 굴절될 때에도 전자는 빛을 방출하는 제동 복사라는 현상을 보인다.\n\n광자와 자유전자가 비탄성적 충돌을 할 때에 콤프턴 산란이 발생한다. 이 때 광자의 운동량과 에너지가 전자에 전달되어 산란된 빛의 파장은 산란각에 따라 늘어나는데, 그 최대치는 로 콤프턴 파장이라고도 불린다. 가시광선처럼 빛의 파장이 마이크로미터 단위를 가진다면 이처럼 작은 차이는 무시될 수 있다. 이처럼 장파장과 전자 사이의 상호작용은 톰슨 산란이라는 별개의 원리로 따로 설명한다."
    # shortened_text = "联邦最高法院在泰勒任内两度出缺，大法官史密斯·汤普森与亨利·鲍德温分别在1843和1844年逝世。泰勒与辉格党控制的参议院关系紧张，提名的约翰·坎菲尔德·斯潘塞、鲁本·沃沃斯、爱德华·金、约翰·里德都被否决，其中沃沃斯否决三次，金两次。参议院此举的重要原因是想让最高法院空缺持续到泰勒下台，克莱赢得1844年大选后再提名人选。泰勒共有四名大法官人选遭参议院否决，比其他美国总统都多。\n\n1845年2月，将在一个月内离任的泰勒提名萨缪尔·内尔森接手汤普森大法官席位。内尔森是以谨慎且毫无争议闻名的民主党法学家，但参议院确认提名仍然出人意料。鲍德温的席位一直空缺到波尔克提名罗伯特·格里尔，参议院1846年确认。"
    LEVEL = "CEFR C2"
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