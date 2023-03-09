1. 文件夹visulization plot下包含EDA过程中用到的可视化的代码：
	- comment_length_distribution_plot.py 统计预测用到的评论在字母级别和单词级别的长度分布。
	- feature_correlation_plot.py 可视化构造的文本特征与target及其他toxicity_subtype的相关性。
	- target_distribution_plot.py 统计训练集上target的分布情况。

2. 文件夹feature generation下包含特征工程过程中用到的提取文本特征的代码：
	- calculate_features.py 生成评论的文本统计特征
	- generate_important_words.py 统计对于预测准确性较为重要的单词

3. 文件夹model train examples下包含Bert和LSTM模型训练和预测的代码：
	- bert_focal_loss.ipynb 包含Bert模型的训练和预测的示例。
	- lstm_focal_loss.ipynb 包含LSTM模型的训练和预测的示例。
	- prediction.py 为使用训练后的Bert模型对测试集进行预测的代码。

	