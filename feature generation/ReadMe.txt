1. generate_importance_words.py 里统计了 toxic 和 good word 的信息。
    
	从中可以得到 toxic word candidates 和 good word candidates 两个列表，以及对其中词的 toxic 和 good 的打分。
	打分包括了：
    
	a. 词出现的频率；
	b. 词覆盖的句子的比例；
	c. 按 toxic 和 non_toxic分两类处理；
	d. 对得分取 log，控制数值的规模。


2. calculate_feature.py 中是计算特征的代码。
	appended_features 中列出了所有计算的特征。包括
    
	'ft_total_length',  
    
	'ft_capitals',                  //大写字母数量  
    
	'ft_caps_vs_length',            //大写字母比例
    
	'ft_num_exclamation_marks',     //感叹号数量
    
	'ft_num_question_marks', 
    
	'ft_num_punctuation', 
    
	'ft_num_symbols', 
    
	'ft_num_words', 
    
	'ft_num_unique_words', 
    
	'ft_words_vs_unique', 
    
	'ft_num_smilies',       //简单笑脸数量
    
	'ft_num_special_punc', 
    
	'ft_num_strange_font',  //特殊字体字符，按统计基本都出现在广告里
    
	'ft_num_toxic_words',   // toxic word 数量
    
	'ft_toxic_words_score', // 句子 toxic 得分
    
	'ft_num_good_words',    // good word 数量
    
	'ft_good_words_score'   // 句子 good 得分



3. 在 toxic_word_candidates.csv 和 good_word_candidates.csv 中列出了每个词的 toxic 和 good 得分。