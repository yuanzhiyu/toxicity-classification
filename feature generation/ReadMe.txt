1. generate_importance_words.py ��ͳ���� toxic �� good word ����Ϣ��
    
	���п��Եõ� toxic word candidates �� good word candidates �����б��Լ������дʵ� toxic �� good �Ĵ�֡�
	��ְ����ˣ�
    
	a. �ʳ��ֵ�Ƶ�ʣ�
	b. �ʸ��ǵľ��ӵı�����
	c. �� toxic �� non_toxic�����ദ��
	d. �Ե÷�ȡ log��������ֵ�Ĺ�ģ��


2. calculate_feature.py ���Ǽ��������Ĵ��롣
	appended_features ���г������м��������������
    
	'ft_total_length',  
    
	'ft_capitals',                  //��д��ĸ����  
    
	'ft_caps_vs_length',            //��д��ĸ����
    
	'ft_num_exclamation_marks',     //��̾������
    
	'ft_num_question_marks', 
    
	'ft_num_punctuation', 
    
	'ft_num_symbols', 
    
	'ft_num_words', 
    
	'ft_num_unique_words', 
    
	'ft_words_vs_unique', 
    
	'ft_num_smilies',       //��Ц������
    
	'ft_num_special_punc', 
    
	'ft_num_strange_font',  //���������ַ�����ͳ�ƻ����������ڹ����
    
	'ft_num_toxic_words',   // toxic word ����
    
	'ft_toxic_words_score', // ���� toxic �÷�
    
	'ft_num_good_words',    // good word ����
    
	'ft_good_words_score'   // ���� good �÷�



3. �� toxic_word_candidates.csv �� good_word_candidates.csv ���г���ÿ���ʵ� toxic �� good �÷֡�