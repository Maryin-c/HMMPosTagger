import numpy as np
from collections import defaultdict, Counter
from itertools import product
import warnings
import time

class HMMPOSTagger_multi_lambda_morph:
    def __init__(self,
                 lambda_interp=0.5,
                 lambda_emit=1.0,
                 # Morphological feature flags
                 use_init_cap=True,
                 use_all_caps=False,
                 use_all_lower=False,
                 use_prefix=True,
                 use_suffix=True,
                 use_digit=True,
                 use_hyphen=True):
        """
        增强版 HMM Tagger：多 λ 平滑 + 丰富形态学特征
        
        Args:
          lambda_interp: 插值平滑系数 α
          lambda_emit:   Add-λ 发射平滑 λₑ
          use_init_cap:  首字母大写特征
          use_all_caps:  全部大写特征
          use_all_lower: 全部小写特征
          use_prefix:    前缀特征（2~4 长度）
          use_suffix:    后缀特征（2~4 长度）
          use_digit:     包含数字特征
          use_hyphen:    包含连字符特征
        """
        # HMM 基本参数
        self.states = []
        self.observations = []
        self.init_prob = {}
        self.trans_prob = {}
        self.emit_prob = {}

        # smoothing
        self.lambda_interp = lambda_interp
        self.lambda_emit = lambda_emit

        # feature flags
        self.use_init_cap = use_init_cap
        self.use_all_caps = use_all_caps
        self.use_all_lower = use_all_lower
        self.use_prefix = use_prefix
        self.use_suffix = use_suffix
        self.use_digit = use_digit
        self.use_hyphen = use_hyphen

        # feature counts & probs
        self.feature_counts = defaultdict(Counter)
        self.feature_probs  = {}

        # unigram tag probs for interpolation
        self.tag_unigram = {}

    def _extract_features(self, word):
        feats = []
        w = word
        # 1. 首字母大写
        if self.use_init_cap and w and w[0].isupper():
            feats.append("feat_init_cap")
        # 2. 全大写
        if self.use_all_caps and w.isupper() and any(c.isalpha() for c in w):
            feats.append("feat_all_caps")
        # 3. 全小写
        if self.use_all_lower and w.islower():
            feats.append("feat_all_lower")
        # 4. 前缀 2-4
        if self.use_prefix and len(w) >= 2:
            L = len(w)
            for l in (2, 3, 4):
                if L >= l:
                    pre = w[:l].lower()
                    feats.append(f"feat_pre_{pre}")
        # 5. 后缀 2-4
        if self.use_suffix and len(w) >= 2:
            L = len(w)
            for l in (2, 3, 4):
                if L >= l:
                    suf = w[-l:].lower()
                    feats.append(f"feat_suf_{suf}")
        # 6. 数字
        if self.use_digit and any(ch.isdigit() for ch in w):
            feats.append("feat_has_digit")
        # 7. 连字符
        if self.use_hyphen and "-" in w:
            feats.append("feat_has_hyphen")
        return feats

    def train(self, tagged_sentences):
        # 统计容器
        tag_counts   = Counter()
        word_tag_cnt = defaultdict(Counter)
        trans_cnt    = defaultdict(Counter)
        init_cnt     = Counter()

        # 1) 统计 HMM 计数 & 特征计数
        for sent in tagged_sentences:
            prev = None
            for idx, (word, tag) in enumerate(sent):
                tag_counts[tag] += 1
                word_tag_cnt[tag][word] += 1
                if idx == 0:
                    init_cnt[tag] += 1
                else:
                    trans_cnt[prev][tag] += 1
                prev = tag

                # 特征计数
                feats = self._extract_features(word)
                for f in feats:
                    self.feature_counts[tag][f] += 1

        # 列表化
        self.states       = list(tag_counts.keys())
        self.observations = list({w for cnt in word_tag_cnt.values() for w in cnt})

        V_tag = len(self.states)
        V_obs = len(self.observations)
        total_tags = sum(tag_counts.values())

        # 2) 初始概率 Add-λ
        tot_init = sum(init_cnt.values()) + self.lambda_emit * V_tag
        for y in self.states:
            self.init_prob[y] = (init_cnt[y] + self.lambda_emit) / tot_init

        # 3) 发射概率 Add-λ
        self.emit_prob = {}
        for y in self.states:
            denom = tag_counts[y] + self.lambda_emit * V_obs
            self.emit_prob[y] = {
                w: (word_tag_cnt[y][w] + self.lambda_emit) / denom
                for w in self.observations
            }

        # 4) 转移概率 插值平滑
        self.tag_unigram = {y: tag_counts[y]/total_tags for y in self.states}
        mle_bigram = {
            y: {y2: trans_cnt[y][y2]/tag_counts[y] if tag_counts[y]>0 else 0.0
                for y2 in self.states}
            for y in self.states
        }
        self.trans_prob = {}
        for y in self.states:
            self.trans_prob[y] = {
                y2: self.lambda_interp * mle_bigram[y][y2]
                    + (1-self.lambda_interp) * self.tag_unigram[y2]
                for y2 in self.states
            }

        # 5) 特征概率 P(f|y)
        self.feature_probs = {}
        for y in self.states:
            total_f = sum(self.feature_counts[y].values())
            denom = total_f if total_f>0 else 1
            self.feature_probs[y] = {
                f: self.feature_counts[y][f] / denom
                for f in self.feature_counts[y]
            }

    def viterbi_decode(self, sentence):
        words = sentence
        T = len(words)
        N = len(self.states)

        dp = np.full((N, T), -np.inf)
        bp = np.zeros((N, T), dtype=int)

        # t=0
        for i, y in enumerate(self.states):
            e = self.emit_prob[y].get(words[0])
            if e is None:
                feats = self._extract_features(words[0])
                if feats:
                    e = np.mean([self.feature_probs[y].get(f, 0.0) for f in feats])
                else:
                    e = 1e-12
            dp[i,0] = np.log(self.init_prob[y] + 1e-12) + np.log(e + 1e-12)

        # t>0
        for t in range(1, T):
            feats_cache = None
            for i, y in enumerate(self.states):
                # emission
                e = self.emit_prob[y].get(words[t])
                if e is None:
                    if feats_cache is None:
                        feats_cache = self._extract_features(words[t])
                    if feats_cache:
                        e = np.mean([self.feature_probs[y].get(f, 0.0) for f in feats_cache])
                    else:
                        e = 1e-12
                log_e = np.log(e + 1e-12)

                # transition + best prev
                scores = [dp[j,t-1] + np.log(self.trans_prob[self.states[j]][y] + 1e-12)
                          for j in range(N)]
                best_j = int(np.argmax(scores))
                dp[i,t]   = scores[best_j] + log_e
                bp[i,t]   = best_j

        # 回溯
        last = int(np.argmax(dp[:,T-1]))
        tags_idx = [last]
        for t in range(T-1, 0, -1):
            last = bp[last,t]
            tags_idx.append(last)
        tags_idx.reverse()
        return [self.states[i] for i in tags_idx]
    
class HMMPOSTagger_trigram:
    def __init__(self, lambda_emit=1.0, lambda_trans=1.0):
        """
        Args:
            lambda_emit: 发射概率的 Add-λ 平滑参数
            lambda_trans: 三元转移概率的 Add-λ 平滑参数
        """
        self.states = []             # 隐状态集合
        self.observations = []       # 观测词汇集合
        self.init_prob = None        # P(y1|<s>,<s>) 和 P(y2|<s>,y1) 的初始概率
        self.trans_prob = None       # 三元转移 P(y_i | y_{i-2}, y_{i-1})
        self.emit_prob = None        # 发射概率 P(w | y)
        self.lambda_emit = lambda_emit
        self.lambda_trans = lambda_trans

    def train(self, tagged_sentences):
        # 计数器
        tag_unigram_cnt   = Counter()
        tag_bigram_cnt    = defaultdict(Counter)
        tag_trigram_cnt   = defaultdict(Counter)
        word_tag_cnt      = defaultdict(Counter)
        # special start tokens
        START = "<s>"

        for sent in tagged_sentences:
            # 在头部添加两个 <s>
            prev2, prev1 = START, START
            tag_unigram_cnt[prev2] += 1
            tag_unigram_cnt[prev1] += 1

            for word, tag in sent:
                # 1) unigram
                tag_unigram_cnt[tag] += 1
                # 2) bigram (for computing P(y2 | <s>, y1) if needed)
                tag_bigram_cnt[prev1][tag] += 1
                # 3) trigram
                tag_trigram_cnt[(prev2, prev1)][tag] += 1
                # 4) emission
                word_tag_cnt[tag][word] += 1

                # shift
                prev2, prev1 = prev1, tag

        # 抽取标签和词汇表
        self.states = list(tag_unigram_cnt.keys())
        self.observations = list({w for cnt in word_tag_cnt.values() for w in cnt})

        V_tag = len(self.states)
        V_word = len(self.observations)

        # —— 发射概率 Add-λ
        self.emit_prob = {}
        for y in self.states:
            total_emit = sum(word_tag_cnt[y].values()) + self.lambda_emit * V_word
            self.emit_prob[y] = {
                w: (word_tag_cnt[y].get(w, 0) + self.lambda_emit) / total_emit
                for w in self.observations
            }

        # —— 初始概率：需要两步
        # P(y1 | <s>,<s>) 以及 P(y2 | <s>, y1)
        # 我们把它们都存到 init_prob[(prev2,prev1)][curr]
        self.init_prob = {}
        # 当 prev2=prev1=<s>, curr=y1
        total0 = sum(tag_bigram_cnt[START].values()) + self.lambda_trans * V_tag
        for y in self.states:
            # 这里使用 bigram_cnt[<s>][y] 作为 y1 的初始分布
            c = tag_bigram_cnt[START].get(y, 0)
            self.init_prob[(START, START), y] = (c + self.lambda_trans) / total0

        # P(y2 | <s>, y1)
        for y1 in self.states:
            total1 = sum(tag_trigram_cnt[(START, y1)].values()) + self.lambda_trans * V_tag
            for y2 in self.states:
                c = tag_trigram_cnt[(START, y1)].get(y2, 0)
                self.init_prob[(START, y1), y2] = (c + self.lambda_trans) / total1

        # —— 三元转移概率 P(y_i | y_{i-2}, y_{i-1})
        self.trans_prob = {}
        for (prev2, prev1), cnts in tag_trigram_cnt.items():
            total = sum(cnts.values()) + self.lambda_trans * V_tag
            self.trans_prob[(prev2, prev1)] = {
                curr: (cnts.get(curr, 0) + self.lambda_trans) / total
                for curr in self.states
            }

    def viterbi_decode(self, words):
        T = len(words)
        N = len(self.states)
        START = "<s>"

        # dp[(i, j, t)]: at position t, last two states are i,j (indices into self.states)
        # 为节省内存，这里我们用一个字典来存储当前 t 步和上一步 t-1 步
        dp_prev = {}
        backpointer = {}

        # ---- t = 0, 只有 y1，y0 都是 <s>,<s> → 下一步会填 y1
        # 但为了统一，我们先“虚拟”填入 dp_prev[(<s>,<s>)] = 0
        dp_prev[(START, START)] = 0.0

        # ---- t = 1, 填 y1
        dp_curr = {}
        for y1 in self.states:
            prob = self.init_prob[(START, START), y1]
            emit = self.emit_prob[y1].get(words[0], 1e-12)
            dp_curr[(START, y1)] = dp_prev[(START, START)] + np.log(prob) + np.log(emit)
            backpointer[(1, (START, y1))] = START
        dp_prev = dp_curr

        # ---- t = 2, 填 y2
        dp_curr = {}
        for (prev2, y1), score in dp_prev.items():
            for y2 in self.states:
                prob = self.init_prob[(prev2, y1), y2]
                emit = self.emit_prob[y2].get(words[1], 1e-12)
                new_score = score + np.log(prob) + np.log(emit)
                dp_curr[(y1, y2)] = new_score
                backpointer[(2, (y1, y2))] = prev2
        dp_prev = dp_curr

        # ---- t >= 3
        for t in range(2, T):
            dp_curr = {}
            for (prev2, prev1), score in dp_prev.items():
                for curr in self.states:
                    trans = self.trans_prob.get((prev2, prev1), {}).get(curr, 1e-12)
                    emit = self.emit_prob[curr].get(words[t], 1e-12)
                    new_score = score + np.log(trans) + np.log(emit)

                    key = (prev1, curr)
                    if new_score > dp_curr.get(key, -np.inf):
                        dp_curr[key] = new_score
                        backpointer[(t+1, key)] = prev2
            dp_prev = dp_curr

        # ---- 终止：选择 t = T 处得分最高的 (y_{T-1}, y_T)
        best_pair, best_score = max(dp_prev.items(), key=lambda x: x[1])

        # 回溯
        tags = [None] * T
        tags[T-1], tags[T-2] = best_pair[1], best_pair[0]
        for t in range(T, 2, -1):
            prev2 = backpointer[(t, (tags[t-2], tags[t-1]))]
            tags[t-3] = prev2

        return tags

class HMMPOSTagger_multi_lambda:
    def __init__(self, lambda_interp=0.5, lambda_emit=1.0):
        """
        Args:
            lambda_interp: 转移概率插值系数（MLE权重）
            lambda_emit: 发射概率的 Add-λ 平滑参数
        """
        self.states = []            # 词性标签集合
        self.observations = []      # 单词集合
        self.init_prob = None       # 初始概率（Add-λ 平滑）
        self.trans_prob = None      # 转移概率（插值平滑）
        self.emit_prob = None       # 发射概率（Add-λ 平滑）
        self.lambda_interp = lambda_interp
        self.lambda_emit = lambda_emit
        self.tag_unigram = None     # 标签的一元概率
    
    def train(self, tagged_sentences):
        # 统计词性标签、单词频次和转移频次
        tag_counts = defaultdict(int)
        word_tag_counts = defaultdict(lambda: defaultdict(int))
        trans_counts = defaultdict(lambda: defaultdict(int))
        init_counts = defaultdict(int)
        
        for sentence in tagged_sentences:
            prev_tag = None
            for idx, (word, tag) in enumerate(sentence):
                tag_counts[tag] += 1
                word_tag_counts[tag][word] += 1
                if idx == 0:
                    init_counts[tag] += 1
                else:
                    trans_counts[prev_tag][tag] += 1
                prev_tag = tag
        
        self.states = list(tag_counts.keys())
        self.observations = list({word for tag_dict in word_tag_counts.values() for word in tag_dict.keys()})
        
        # ============== 初始概率（Add-λ 平滑） ==============
        total_init = sum(init_counts.values()) + self.lambda_emit * len(self.states)
        self.init_prob = {
            tag: (init_counts.get(tag, 0) + self.lambda_emit) / total_init 
            for tag in self.states
        }
        
        # ============== 转移概率（插值平滑） ==============
        # 计算标签的一元概率
        total_tags = sum(tag_counts.values())
        self.tag_unigram = {
            tag: count / total_tags 
            for tag, count in tag_counts.items()
        }
        
        # 计算转移概率的 MLE 估计和插值平滑
        self.trans_prob = {}
        for from_tag in self.states:
            total_trans = sum(trans_counts[from_tag].values())
            mle_trans = {
                to_tag: trans_counts[from_tag].get(to_tag, 0) / total_trans 
                if total_trans > 0 else 0
                for to_tag in self.states
            }
            self.trans_prob[from_tag] = {
                to_tag: self.lambda_interp * mle_trans[to_tag] + 
                        (1 - self.lambda_interp) * self.tag_unigram[to_tag]
                for to_tag in self.states
            }
        
        # ============== 发射概率（Add-λ 平滑） ==============
        self.emit_prob = {}
        for tag in self.states:
            total_emit = sum(word_tag_counts[tag].values()) + self.lambda_emit * len(self.observations)
            self.emit_prob[tag] = {
                word: (word_tag_counts[tag].get(word, 0) + self.lambda_emit) / total_emit 
                for word in self.observations
            }
    
    def viterbi_decode(self, sentence):
        """Viterbi算法解码：输入未标注句子，输出最优词性序列"""
        # 转换为单词列表
        words = [word for word in sentence]
        num_tags = len(self.states)
        viterbi = np.zeros((num_tags, len(words)))  # 动态规划表
        backpointers = np.zeros((num_tags, len(words)), dtype=int)  # 路径回溯
        
        # 初始化第一步
        for i, tag in enumerate(self.states):
            emit = self.emit_prob[tag].get(words[0], 1e-10)  # 处理未登录词（极小概率）
            viterbi[i, 0] = np.log(self.init_prob[tag]) + np.log(emit)
        
        # 递推填充表格
        for t in range(1, len(words)):
            for i, curr_tag in enumerate(self.states):
                max_score = -np.inf
                best_prev = 0
                for j, prev_tag in enumerate(self.states):
                    # 转移概率 + 前一步最优得分
                    score = viterbi[j, t-1] + np.log(self.trans_prob[prev_tag].get(curr_tag, 1e-10))
                    if score > max_score:
                        max_score = score
                        best_prev = j
                # 更新当前得分（加上发射概率）
                emit = self.emit_prob[curr_tag].get(words[t], 1e-10)
                viterbi[i, t] = max_score + np.log(emit)
                backpointers[i, t] = best_prev
        
        # 回溯找到最优路径
        best_path = []
        last_step = np.argmax(viterbi[:, -1])
        best_path.append(last_step)
        for t in range(len(words)-1, 0, -1):
            last_step = backpointers[last_step, t]
            best_path.insert(0, last_step)
        
        # 转换为词性标签
        return [self.states[idx] for idx in best_path]

class HMMPOSTagger_interp:
    def __init__(self, lambda_interp=0.5):
        """
        Args:
            lambda_interp: 插值平滑系数，控制转移概率中 MLE 和一元概率的权重
        """
        self.states = []            # 词性标签集合
        self.observations = []      # 单词集合
        self.init_prob = None       # 初始概率
        self.trans_prob = None      # 转移概率（使用插值平滑）
        self.emit_prob = None       # 发射概率（保持 Add-1 平滑）
        self.lambda_interp = lambda_interp  # 插值系数
        self.tag_unigram = None     # 标签的一元概率
    
    def train(self, tagged_sentences):
        # 统计词性标签、单词频次和转移频次
        tag_counts = defaultdict(int)
        word_tag_counts = defaultdict(lambda: defaultdict(int))
        trans_counts = defaultdict(lambda: defaultdict(int))
        init_counts = defaultdict(int)
        
        for sentence in tagged_sentences:
            prev_tag = None
            for idx, (word, tag) in enumerate(sentence):
                tag_counts[tag] += 1
                word_tag_counts[tag][word] += 1
                if idx == 0:
                    init_counts[tag] += 1
                else:
                    trans_counts[prev_tag][tag] += 1
                prev_tag = tag
        
        self.states = list(tag_counts.keys())
        self.observations = list({word for tag_dict in word_tag_counts.values() for word in tag_dict.keys()})
        
        # ============== 初始概率（Add-1 平滑） ==============
        total_init = sum(init_counts.values()) + len(self.states)
        self.init_prob = {
            tag: (init_counts.get(tag, 0) + 1) / total_init 
            for tag in self.states
        }
        
        # ============== 转移概率（插值平滑） ==============
        # 计算标签的一元概率
        total_tags = sum(tag_counts.values())
        self.tag_unigram = {
            tag: count / total_tags 
            for tag, count in tag_counts.items()
        }
        
        # 计算转移概率的 MLE 估计和插值平滑
        self.trans_prob = {}
        for from_tag in self.states:
            total_trans = sum(trans_counts[from_tag].values())
            # MLE 估计
            mle_trans = {
                to_tag: trans_counts[from_tag].get(to_tag, 0) / total_trans 
                if total_trans > 0 else 0
                for to_tag in self.states
            }
            # 插值平滑
            self.trans_prob[from_tag] = {
                to_tag: self.lambda_interp * mle_trans[to_tag] + 
                        (1 - self.lambda_interp) * self.tag_unigram[to_tag]
                for to_tag in self.states
            }
        
        # ============== 发射概率（Add-1 平滑） ==============
        self.emit_prob = {}
        for tag in self.states:
            total_emit = sum(word_tag_counts[tag].values()) + len(self.observations)
            self.emit_prob[tag] = {
                word: (word_tag_counts[tag].get(word, 0) + 1) / total_emit 
                for word in self.observations
            }
    
    def viterbi_decode(self, sentence):
        """Viterbi算法解码：输入未标注句子，输出最优词性序列"""
        # 转换为单词列表
        words = [word for word in sentence]
        num_tags = len(self.states)
        viterbi = np.zeros((num_tags, len(words)))  # 动态规划表
        backpointers = np.zeros((num_tags, len(words)), dtype=int)  # 路径回溯
        
        # 初始化第一步
        for i, tag in enumerate(self.states):
            emit = self.emit_prob[tag].get(words[0], 1e-10)  # 处理未登录词（极小概率）
            viterbi[i, 0] = np.log(self.init_prob[tag]) + np.log(emit)
        
        # 递推填充表格
        for t in range(1, len(words)):
            for i, curr_tag in enumerate(self.states):
                max_score = -np.inf
                best_prev = 0
                for j, prev_tag in enumerate(self.states):
                    # 转移概率 + 前一步最优得分
                    score = viterbi[j, t-1] + np.log(self.trans_prob[prev_tag].get(curr_tag, 1e-10))
                    if score > max_score:
                        max_score = score
                        best_prev = j
                # 更新当前得分（加上发射概率）
                emit = self.emit_prob[curr_tag].get(words[t], 1e-10)
                viterbi[i, t] = max_score + np.log(emit)
                backpointers[i, t] = best_prev
        
        # 回溯找到最优路径
        best_path = []
        last_step = np.argmax(viterbi[:, -1])
        best_path.append(last_step)
        for t in range(len(words)-1, 0, -1):
            last_step = backpointers[last_step, t]
            best_path.insert(0, last_step)
        
        # 转换为词性标签
        return [self.states[idx] for idx in best_path]

class HMMPOSTagger_add_lambda:
    def __init__(self, lambda_val=1.0):
        """
        Args:
            lambda_val: 自定义平滑参数 λ（默认值为 1，即 Add-1 平滑）
        """
        self.states = []          # 词性标签集合（隐状态）
        self.observations = []    # 单词集合（观测）
        self.init_prob = None     # 初始概率向量
        self.trans_prob = None    # 转移概率矩阵
        self.emit_prob = None     # 发射概率矩阵
        self.lambda_val = lambda_val  # 自定义平滑参数 λ
    
    def train(self, tagged_sentences):
        """训练HMM模型：统计初始概率、转移概率、发射概率（使用 Add-λ 平滑）"""
        # 统计词性标签和单词的出现频次
        tag_counts = defaultdict(int)
        word_tag_counts = defaultdict(lambda: defaultdict(int))
        trans_counts = defaultdict(lambda: defaultdict(int))
        init_counts = defaultdict(int)
        
        # 遍历标注好的句子（每个句子是 (word, tag) 的列表）
        for sentence in tagged_sentences:
            prev_tag = None
            for idx, (word, tag) in enumerate(sentence):
                # 统计词性标签和单词
                tag_counts[tag] += 1
                word_tag_counts[tag][word] += 1
                # 统计转移频次（当前标签到下一个标签）
                if idx == 0:
                    init_counts[tag] += 1
                else:
                    trans_counts[prev_tag][tag] += 1
                prev_tag = tag
        
        # 转换为列表形式（方便索引）
        self.states = list(tag_counts.keys())
        self.observations = list({word for tag_dict in word_tag_counts.values() for word in tag_dict.keys()})
        
        # ============== 使用 Add-λ 平滑计算概率 ==============
        # 计算初始概率
        total_init = sum(init_counts.values()) + self.lambda_val * len(self.states)
        self.init_prob = {
            tag: (init_counts.get(tag, 0) + self.lambda_val) / total_init 
            for tag in self.states
        }
        
        # 计算转移概率
        self.trans_prob = {}
        for from_tag in self.states:
            total_trans = sum(trans_counts[from_tag].values()) + self.lambda_val * len(self.states)
            self.trans_prob[from_tag] = {
                to_tag: (trans_counts[from_tag].get(to_tag, 0) + self.lambda_val) / total_trans 
                for to_tag in self.states
            }
        
        # 计算发射概率
        self.emit_prob = {}
        for tag in self.states:
            total_emit = sum(word_tag_counts[tag].values()) + self.lambda_val * len(self.observations)
            self.emit_prob[tag] = {
                word: (word_tag_counts[tag].get(word, 0) + self.lambda_val) / total_emit 
                for word in self.observations
            }
    
    def viterbi_decode(self, sentence):
        """Viterbi算法解码：输入未标注句子，输出最优词性序列"""
        # 转换为单词列表
        words = [word for word in sentence]
        num_tags = len(self.states)
        viterbi = np.zeros((num_tags, len(words)))  # 动态规划表
        backpointers = np.zeros((num_tags, len(words)), dtype=int)  # 路径回溯
        
        # 初始化第一步
        for i, tag in enumerate(self.states):
            emit = self.emit_prob[tag].get(words[0], 1e-10)  # 处理未登录词（极小概率）
            viterbi[i, 0] = np.log(self.init_prob[tag]) + np.log(emit)
        
        # 递推填充表格
        for t in range(1, len(words)):
            for i, curr_tag in enumerate(self.states):
                max_score = -np.inf
                best_prev = 0
                for j, prev_tag in enumerate(self.states):
                    # 转移概率 + 前一步最优得分
                    score = viterbi[j, t-1] + np.log(self.trans_prob[prev_tag].get(curr_tag, 1e-10))
                    if score > max_score:
                        max_score = score
                        best_prev = j
                # 更新当前得分（加上发射概率）
                emit = self.emit_prob[curr_tag].get(words[t], 1e-10)
                viterbi[i, t] = max_score + np.log(emit)
                backpointers[i, t] = best_prev
        
        # 回溯找到最优路径
        best_path = []
        last_step = np.argmax(viterbi[:, -1])
        best_path.append(last_step)
        for t in range(len(words)-1, 0, -1):
            last_step = backpointers[last_step, t]
            best_path.insert(0, last_step)
        
        # 转换为词性标签
        return [self.states[idx] for idx in best_path]

def evaluate(tagger, tagged_sentences):
    """评估模型在带标签的数据集上的准确率"""
    total_correct = 0
    total_words = 0
    
    for sentence in tagged_sentences:
        # 从带标签的句子中分离出单词和真实词性
        words = [word for word, tag in sentence]
        true_tags = [tag for word, tag in sentence]
        
        # 使用模型预测词性
        pred_tags = tagger.viterbi_decode(words)
        
        # 确保预测和真实标签长度一致
        if len(pred_tags) != len(true_tags):
            raise ValueError("预测标签与真实标签长度不一致")
        
        # 统计正确数
        for pred, true in zip(pred_tags, true_tags):
            if pred == true:
                total_correct += 1
        total_words += len(true_tags)
    
    accuracy = total_correct / total_words if total_words > 0 else 0.0
    return accuracy

def read_data(path):
    sentences = []
    sentence = []
    with open(path) as f:
        for line in f:
            word, label = line.strip().split('/')
            if word == "###":
                sentences.append(sentence)
                sentence = []
            else:
                sentence.append((word, label))
    return sentences


def cross_validate_add_lambda(train_data, valid_data, lambda_candidates=[0, 0.001, 0.01, 0.05, 0.1, 0.5, 1]):
    best_accuracy = 0
    best_lambda = 1.0
    for lambda_val in lambda_candidates:
        tagger = HMMPOSTagger_add_lambda(lambda_val=lambda_val)
        tagger.train(train_data)
        accuracy = evaluate(tagger, valid_data)  # 自定义评估函数
        print(f"lambda: {lambda_val}, acc:{accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lambda = lambda_val
    return best_lambda

def cross_validate_interp(train_data, valid_data, lambda_candidates=[0.2, 0.25, 0.3, 0.35, 0.4]):
    """交叉验证选择最佳 lambda_interp"""
    best_lambda = 0.5
    best_accuracy = 0
    for lambda_val in lambda_candidates:
        model = HMMPOSTagger_interp(lambda_interp=lambda_val)
        model.train(train_data)
        accuracy = evaluate(model, valid_data)  # 需实现评估函数
        print(f"lambda: {lambda_val}, acc:{accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lambda = lambda_val
    return best_lambda

def cross_validate_multi_lambda(train_data, valid_data, 
                         lambda_interp_candidates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1],
                         lambda_emit_candidates=[0, 0.001, 0.01, 0.1, 0.5, 1]):
    """交叉验证选择最佳 lambda_interp 和 lambda_emit"""
    best_params = (0.5, 1.0)
    best_accuracy = 0
    
    results = []
    # 遍历所有参数组合
    for lambda_interp, lambda_emit in product(lambda_interp_candidates, lambda_emit_candidates):
        model = HMMPOSTagger_multi_lambda(lambda_interp=lambda_interp, lambda_emit=lambda_emit)
        model.train(train_data)
        accuracy = evaluate(model, valid_data)  # 需实现评估函数
        results.append({
                "lambda_interp": lambda_interp,
                "lambda_emit": lambda_emit,
                "accuracy": accuracy # 保留4位小数
            })
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (lambda_interp, lambda_emit)
    
    sorted_results = sorted(results, key=lambda x: -x["accuracy"])
    # 打印排序结果
    print("\n===== 交叉验证结果排序 =====")
    print("Rank | λ_interp | λ_emit | Accuracy")
    for idx, res in enumerate(sorted_results, 1):
        print(f"{idx} | {res['lambda_interp']} | {res['lambda_emit']} | {res['accuracy']}")

    return best_params

def cross_validate_trigram(train_data, valid_data,
                           lambda_trans_candidates=[0.01, 0.1, 0.5, 1.0],
                           lambda_emit_candidates=[0.01, 0.1, 0.5, 1.0]):
    """
    对 HMMPOSTagger_trigram 进行交叉验证，选择最佳 lambda_trans 和 lambda_emit。

    Args:
        train_data:    训练集，格式为 List[List[(word, tag)]]
        valid_data:    验证集，同 train_data 格式
        lambda_trans_candidates:  三元转移平滑参数候选列表
        lambda_emit_candidates:   发射平滑参数候选列表

    Returns:
        best_params: (best_lambda_trans, best_lambda_emit)
    """
    best_params = (lambda_trans_candidates[0], lambda_emit_candidates[0])
    best_acc = 0.0
    results = []

    for lambda_trans, lambda_emit in product(lambda_trans_candidates, lambda_emit_candidates):
        # 初始化并训练模型
        model = HMMPOSTagger_trigram(lambda_emit=lambda_emit,
                                     lambda_trans=lambda_trans)
        model.train(train_data)

        # 在验证集上评估
        acc = evaluate(model, valid_data)  # 你需要实现或导入 evaluate(model, data) → accuracy
        results.append({
            "lambda_trans": lambda_trans,
            "lambda_emit":  lambda_emit,
            "accuracy":     acc
        })
        print(results[len(results) - 1])

        if acc > best_acc:
            best_acc = acc
            best_params = (lambda_trans, lambda_emit)

    # 按准确率从高到低排序并打印
    results.sort(key=lambda x: -x["accuracy"])
    print("\n===== Trigram HMM Cross-Validation Results =====")
    print(" Rank | λ_trans | λ_emit | Accuracy ")
    print("------+---------+--------+----------")
    for rank, res in enumerate(results, 1):
        print(f"{rank:4d} | {res['lambda_trans']:7.3f} | {res['lambda_emit']:6.3f} | {res['accuracy']:.4f}")

    print(f"\n>> Best params: λ_trans = {best_params[0]}, λ_emit = {best_params[1]}  (acc = {best_acc:.4f})")
    return best_params

# def cross_validate_multi_lambda_morphology(train_data, valid_data, 
#                          lambda_interp_candidates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1],
#                          lambda_emit_candidates=[0, 0.001, 0.01, 0.1, 0.5, 1]):
#     """交叉验证选择最佳 lambda_interp 和 lambda_emit"""
#     best_params = (0.5, 1.0)
#     best_accuracy = 0
    
#     results = []
#     # 遍历所有参数组合
#     for lambda_interp, lambda_emit in product(lambda_interp_candidates, lambda_emit_candidates):
#         model = HMMPOSTagger_multi_lambda_morph(lambda_interp=lambda_interp, lambda_emit=lambda_emit)
#         model.train(train_data)
#         accuracy = evaluate(model, valid_data)  # 需实现评估函数
#         results.append({
#                 "lambda_interp": lambda_interp,
#                 "lambda_emit": lambda_emit,
#                 "accuracy": accuracy # 保留4位小数
#             })
#         print(results[len(results) - 1])
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_params = (lambda_interp, lambda_emit)
    
#     sorted_results = sorted(results, key=lambda x: -x["accuracy"])
#     # 打印排序结果
#     print("\n===== 交叉验证结果排序 =====")
#     print("Rank | λ_interp | λ_emit | Accuracy")
#     for idx, res in enumerate(sorted_results, 1):
#         print(f"{idx} | {res['lambda_interp']} | {res['lambda_emit']} | {res['accuracy']}")

#     return best_params

def cross_validate_multi_lambda_morphology(
    train_data,
    valid_data,
    lambda_interp_candidates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0],
    lambda_emit_candidates=[0.0, 0.001, 0.01, 0.1, 0.5, 1.0]
):
    """
    对 HMMPOSTagger_multi_lambda_morph 进行全组合交叉验证，
    包括平滑参数和所有形态学特征开关，总共 2^7 * len(lambda_interp) * len(lambda_emit) 次实验。
    记录并输出训练集和验证集准确率，但排序仅基于验证集。

    Returns:
        best_params: dict，包含最优配置及其 train_accuracy, valid_accuracy
        sorted_results: 排序后的所有结果列表
    """
    flag_names = [
        "use_init_cap",
        "use_all_caps",
        "use_all_lower",
        "use_prefix",
        "use_suffix",
        "use_digit",
        "use_hyphen"
    ]

    # 预计算总实验次数
    num_flags = 2 ** len(flag_names)
    num_interps = len(lambda_interp_candidates)
    num_emits = len(lambda_emit_candidates)
    total_jobs = num_flags * num_interps * num_emits

    start_time = time.time()
    job_count = 0

    best_params = None
    best_valid_acc = 0.0
    all_results = []

    # 生成所有布尔特征组合
    flag_values = list(product([False, True], repeat=len(flag_names)))

    for lambda_interp, lambda_emit in product(lambda_interp_candidates, lambda_emit_candidates):
        for flags in flag_values:
            job_count += 1

            # 构造模型参数字典
            params = {
                "lambda_interp": lambda_interp,
                "lambda_emit": lambda_emit
            }
            params.update({name: flag for name, flag in zip(flag_names, flags)})

            # 训练
            model = HMMPOSTagger_multi_lambda_morph(**params)
            model.train(train_data)

            # 评估训练和验证准确率
            # train_acc = evaluate(model, train_data)
            valid_acc = evaluate(model, valid_data)

            # 保存结果
            row = dict(params)
            # row["train_accuracy"] = round(train_acc, 4)
            row["valid_accuracy"] = valid_acc
            all_results.append(row)

            # 计算 ETA
            elapsed = time.time() - start_time
            avg_time = elapsed / job_count
            remaining = avg_time * (total_jobs - job_count)

            # 实时打印并立即 flush
            print(
                f"[{job_count}/{total_jobs}] "
                + f"interp={lambda_interp}, emit={lambda_emit}, "
                + ", ".join(f"{name}={flag}" for name, flag in zip(flag_names, flags))
                + f" → valid_acc={row['valid_accuracy']} | ETA: {remaining:.1f}s",
                flush=True
            )

            # 更新最优（基于验证集）
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_params = row.copy()

    # 按验证集准确率降序排序
    sorted_results = sorted(all_results, key=lambda x: -x["valid_accuracy"])

    # 打印汇总排名，并立即 flush
    print("\n===== 交叉验证结果排序（按验证集准确率） =====", flush=True)
    header = ["Rank", "λ_interp", "λ_emit"] + flag_names + ["Valid_Acc"]
    print(" | ".join(header), flush=True)
    print("-" * (len(header) * 12), flush=True)
    for idx, res in enumerate(sorted_results, 1):
        vals = [
            str(idx),
            str(res["lambda_interp"]),
            str(res["lambda_emit"])
        ]
        vals += [str(res[name]) for name in flag_names]
        vals += [f"{res['valid_accuracy']}"]
        print(" | ".join(vals), flush=True)

    return best_params, sorted_results

def train_and_test(entrain, endev, entest):
    train_data = read_data(entrain)
    valid_data = read_data(endev)
    test_data = read_data(entest)

    # print(f"add lambda, best lambda: {cross_validate_add_lambda(train_data, valid_data)}")
    # print(f"interp, best lambda: {cross_validate_interp(train_data, valid_data)}")
    # print(f"multi, best lambda: {cross_validate_multi_lambda(train_data, valid_data)}")
    # print(f"multi, best lambda: {cross_validate_trigram(train_data, valid_data)}")
    print(f"multi morphology, best lambda: {cross_validate_multi_lambda_morphology(train_data, valid_data)}", flush=True)
    
    # # 初始化并训练模型
    # tagger = HMMPOSTagger()
    # tagger.train(train_data)

    # # 评估验证集和测试集
    # train_accuracy = evaluate(tagger, train_data)
    # valid_accuracy = evaluate(tagger, valid_data)
    # test_accuracy = evaluate(tagger, test_data)
    
    # print(f"训练集准确率: {train_accuracy:.4f}")
    # print(f"验证集准确率: {valid_accuracy:.4f}")
    # print(f"测试集准确率: {test_accuracy:.4f}")

# train_and_test("./data/ictrain", "./data/ictest", "./data/ictest")
train_and_test("./data/entrain", "./data/endev", "./data/endev")