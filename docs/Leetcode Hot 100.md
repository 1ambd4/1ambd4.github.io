---
author: 紫薯布丁
date: 2024-03-22
categories: Leetcode
tags: ["Leetcode"]
---

# Leetcode Hot 100

一天刷完 Hot100，好耶，:)

## 哈希

### 1. 两数之和

注意一下同一个数字不能用两次，其他没什么好说的了。

```cpp {28}
vector<int> twoSum(vector<int>& nums, int target) {
    int n = nums.size();

    vector<array<int, 2>> inums;
    for (int i = 0; i < n; ++i) inums.push_back({i, nums[i]});

    std::sort(inums.begin(), inums.end(), [](array<int, 2>& a, array<int, 2>& b) -> bool {
        return a[1] < b[1];
    });

    auto search = [&inums, n](int t) {
        int l = 0;
        int r = n - 1;

        while (l < r) {
            int m = (l + r + 1)>> 1;
            if (inums[m][1] <= t) l = m;
            else r = m - 1;
        }

        if (inums[l][1] == t) return l;
        else return -1;
    };

    for (int i = 0; i < n; ++i) {
        int t = target - inums[i][1];
        int j = search(t);
        if (j != -1 && j != i) return {inums[i][0], inums[j][0]};
    }

    return {};
}
```

### 49. 字母异位词分组

```cpp
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    vector<vector<string>> res;

    unordered_map<string, vector<string>> m;

    for (const auto& s : strs) {
        string mask = string(26, static_cast<char>(0));
        for (char c : s) {
            c -= 'a';
            mask[c]++;
        }
        m[mask].push_back(s);
    }

    for (const auto& [k, v] : m) {
        res.push_back(vector<string>());
        for (const auto& s : v) {
            res.back().push_back(s);
        }
    }

    return res;
}
```

### 128. 最长连续序列

`m[i]` 用来记录 `nums[i]` 所在的连续序列的长度，`m[i] == 0` 表示该点尚未被访问过。那么 `m[i-1]` 和 `m[i+1]` 分别表示 `nums[i]` 左右相邻数字所在的连续序列的长度，左边加右边加上自身的长度 1，那么就可以计算出 `nums[i]` 所在的连续序列的长度了。当然，需要注意的是，要将连续序列的长度信息进行更新，并且只需要更新两端就可以了，因为是连续序列，所以每次只会访问到距离为 $1$ 的节点，序列中间的节点是不可能被访问到的。最后再将自身的长度进行更新（这里主要是为了标记已访问以及处理左右相邻数字当前不存在的情况）。

```cpp {11-13}
int longestConsecutive(vector<int>& nums) {
    unordered_map<int, int> m;
    int res = 0;

    for (const auto& x : nums) {
        if (m[x] == 0) {
            int l = m[x-1];
            int r = m[x+1];
            int d = l + 1 + r;

            m[x-l] = d;
            m[x+r] = d;
            m[x] = d;

            res = std::max(res, d);
        }
    }

    return res;
}
```

## 双指针

### 283. 移动零

```cpp
void moveZeroes(vector<int>& nums) {
    int n = nums.size();
    int l = 0;
    int r = 0;

    while (r < n) {
        if (nums[r] != 0) {
            std::swap(nums[l], nums[r]);
            ++l;
        }
        ++r;
    }
}
```

### 11. 盛最多水的容器

```cpp
int maxArea(vector<int>& h) {
    int n = h.size();
    int l = 0;
    int r = n - 1;
    int res = 0;

    while (l < r) {
        int hl = h[l];
        int hr = h[r];
        res = std::max(res, std::min(hl, hr) * (r - l));

        if (hl < hr) ++l;
        else --r;
    }

    return res;
}
```

### 15. 三数之和

注意去重在遍历的时候就做好，会方便很多，别加到结果集之后再处理。

```cpp {26,28}
vector<vector<int>> threeSum(vector<int>& nums) {
    int n  = nums.size();
    vector<vector<int>> res;
    if (n == 0) return res;

    std::sort(nums.begin(), nums.end(), std::less<int>());

    auto search = [&nums, n](int t) -> int {
        int l = 0;
        int r = n - 1;
        while (l < r) {
            int m = (l + r + 1) >> 1;
            if (nums[m] <= t) l = m;
            else r = m - 1;
        }

        if (nums[l] == t) return l;
        else return -1;
    };

    for (int i = 0, j = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            int t = 0 - nums[i] - nums[j];
            int k = search(t);
            if (k != -1 && k > j) res.push_back({nums[i], nums[j], nums[k]});
            while (j+1 < n && nums[j+1] == nums[j]) ++j;
        }
        while (i+1 < n && nums[i+1] == nums[i]) ++i;
    }

    return res;
}
```

### 42. 接雨水

```cpp
int trap(vector<int>& h) {
    int n = h.size();
    int l = 0;
    int r = n - 1;
    int ml = 0;
    int mr = 0;

    int res = 0;

    while (l < r) {
        ml = std::max(ml, h[l]);
        mr = std::max(mr, h[r]);

        if (ml < mr) {
            res += ml - h[l];
            ++l;
        } else {
            res += mr - h[r];
            --r;
        }
    }

    return res;
}
```

## 滑动窗口

### 3. 无重复字符的最长子串

```cpp
int lengthOfLongestSubstring(string s) {
    int n = s.size();
    int l = 0;
    int r = 0;

    unordered_map<char, int> wind;

    int res = 0;

    while(r < n) {
        char c = s[r];
        wind[c]++;
        ++r;

        while (wind[c] > 1) {
            char t = s[l];
            wind[t]--;
            ++l;
        }

        res = std::max(res, r - l);
    }

    return res;
}
```

### 438. 找到字符串中所有字母异位词

```cpp
vector<int> findAnagrams(string s, string p) {
    int n = s.size();
    int m = p.size();
    int l = 0;
    int r = 0;
    int v = 0;
    vector<int> res;

    unordered_map<char, int> wind;
    unordered_map<char, int> need;
    for (char c : p) need[c]++;

    while (r < n) {
        char c = s[r];
        if (need.count(c)) {
            wind[c]++;
            if (wind[c] == need[c]) ++v;
        }
        ++r;

        while (r - l == p.size()) {
            if (v == need.size()) res.push_back(l);

            char t = s[l];
            if (need.count(t)) {
                if (wind[t] == need[t]) --v;
                wind[t]--;
            }
            ++l;
        }
    }

    return res;
}
```

## 子串

### 560. 和为 K 的子数组

腾讯某次一面算法题，`m[i]` 是前缀和，那么如果求中间某一段的子数组的和公式为: `m[r] - m[l] = k`，所以 11 行才是 `res += m[sum - k]`。

```cpp {11}
int subarraySum(vector<int>& nums, int k) {
    int n = nums.size();

    unordered_map<int, int> m;
    m[0] = 1;

    int res = 0;
    int sum = 0;
    for (const auto& num : nums) {
        sum += num;
        res += m[sum - k];
        m[sum]++;
    }

    return res;
}
```

### 239. 滑动窗口最大值

比较简单的思路是维护一个优先队列，然后每次更新结果的时候，先把不再窗口内的都先出队，此时取出队头即使窗口内的最大值。

```cpp
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    int n = nums.size();
    vector<int> res;

    std::priority_queue<std::pair<int, int>> q;
    for (int i = 0; i < k; ++i) q.push({nums[i], i});

    res.push_back(q.top().first);
    for (int i = k; i < n; ++i) {
        q.push({nums[i], i});
        while (q.top().second <= i - k) q.pop();
        res.push_back(q.top().first);
    }

    return res;
}
```

当然，不想用优先队列的话，可以用普通的队列。

```cpp
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    int n = nums.size();
    std::deque<int> q;

    for (int i = 0; i < k; ++i) {
        while (!q.empty() && nums[i] >= nums[q.back()]) {
            q.pop_back();
        }
        q.push_back(i);
    }

    vector<int> res { nums[q.front()] };
    for (int i = k; i < n; ++i) {
        while (!q.empty() && nums[i] >= nums[q.back()]) {
            q.pop_back();
        }
        q.push_back(i);
        while (!q.empty() && q.front() <= i - k) {
            q.pop_front();
        }
        res.push_back(nums[q.front()]);
    }
    return res;
}
```

### 76. 最小覆盖子串

```cpp
string minWindow(string s, string q) {
    int n = s.size();
    int m = q.size();

    int l = 0;
    int r = 0;
    int v = 0;
    int res = 0;
    int len = n + 1;

    unordered_map<char, int> wind;
    unordered_map<char, int> need;
    for (char c : q) need[c]++;

    while (r < n) {
        char c = s[r];
        if (need.count(c)) {
            wind[c]++;
            if (wind[c] == need[c]) ++v;
        }
        ++r;

        while (v == need.size()) {

            if (r - l < len) {
                res = l;
                len = r - l;
            }

            char t = s[l];
            if (need.count(t)) {
                if (wind[t] == need[t]) --v;
                wind[t]--;
            }
            ++l;
        }
    }

    return len != n + 1 ? s.substr(res, len) : "";
}
```

## 普通数组

### 53. 最大子数组和

```cpp
int maxSubArray(vector<int>& nums) {
    int n = nums.size();
    vector<int> f(n, 0);

    f[0] = nums[0];
    int res = f[0];
    for (int i = 1; i < n; ++i) {
        f[i] = std::max(f[i-1] + nums[i], nums[i]);
        res = std::max(res, f[i]);
    }
    return res;
}
```

### 56. 合并区间

注意 `i` 随着 `j` 在变化，因此需要把 `j` 定义在外层循环里，以把 `j` 的值传递出来。另外，`i == n-1` 和 `j = n-1` 的情况需要特殊处理一下。

```cpp {9,13,22}
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    int n = intervals.size();
    vector<vector<int>> res;

    std::sort(intervals.begin(), intervals.end(), [](vector<int>& a, vector<int>& b) -> bool {
        return a[0] < b[0];
    });

    for (int i = 0, j = 0; i < n; i = j) {
        int l = intervals[i][0];
        int r = intervals[i][1];

        if (i == n - 1) res.push_back({l, r});

        for (j = i + 1; j < n; ++j) {
            int u = intervals[j][0];
            int v = intervals[j][1];

            if (l <= u && u <= r) {
                l = std::min(l, u);
                r = std::max(r, v);
                if (j == n - 1) res.push_back({l, r});
            } else {
                res.push_back({l, r});
                break;
            }
        }
    }

    return res;
}
```

### 189. 轮转数组

比较有意思的一个题目了，$[A, B]_{T}$ => $[A_{T}, B_{T}]$，即先将整体反转，然后再局部反转。

```cpp
void rotate(vector<int>& nums, int k) {
    int n = nums.size();

    auto reverse = [&nums](int l, int r) -> void {
        while (l < r) {
            std::swap(nums[l], nums[r]);
            ++l;
            --r;
        }
    };

    k %= n;
    reverse(0, n - 1);
    reverse(0, k - 1);
    reverse(k, n - 1);
}
```

### 238. 除自身以外数组的乘积

```cpp
vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    int pre = 1;
    int suf = 1;
    vector<int> res(n, 1);

    for (int i = 0; i < n; ++i) {
        res[i] *= pre;
        pre *= nums[i];
    }
    for (int i = n - 1; i >= 0; --i) {
        res[i] *= suf;
        suf *= nums[i];
    }

    return res;
}
```

### 41. 缺失的第一个正数

这题也有点儿意思，原地哈希，`nums[i]` 应该放到下标为 `nums[i] - 1` 的地方，这也就是代码里的 `idx = nums[i] - 1`，遍历每一个元素，不断的将当前元素放到其该处于的位置，一直交换到当前位置的元素归位。但若当前元素和其应该处于的位置上的元素相同，说明出现了重复数字，此时跳出循环，遍历下一个元素即可。当然，在这过程中，如果遇到了不再 $[1, n]$ 范围内的数字，也直接跳出循环，处理下一个元素。那么，最终，没在应该处于的位置上的元素就是缺失的那个。

```cpp {7}
int firstMissingPositive(vector<int>& nums) {
    int n = nums.size();

    for (int i = 0; i < n; ++i) {
        while (nums[i] != i + 1) {
            if (nums[i] <= 0 || nums[i] > n) break;
            int idx = nums[i] - 1;
            if (nums[idx] == nums[i]) break;
            std::swap(nums[i], nums[idx]);
        }
    }

    for (int i = 0; i < n; ++i) {
        if (nums[i] != i + 1) return i + 1;
    }

    return n + 1;
}
```

## 矩阵

### 73. 矩阵置零

```cpp
void setZeroes(vector<vector<int>>& matrix) {
    int m = matrix.size();
    int n = matrix[0].size();
    vector<int> row(m, 0);
    vector<int> col(n, 0);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (matrix[i][j] == 0) {
                row[i] = 1;
                col[j] = 1;
            }
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (row[i] == 1 || col[j] == 1) {
                matrix[i][j] = 0;
            }
        }
    }
}
```

### 54. 螺旋矩阵

这题比较好的方法是动态的调整上下边界。

```cpp
vector<int> spiralOrder(vector<vector<int>>& matrix) {
    vector<int> res;
    if (matrix.empty()) return res;

    // 动态调整上下左右边界
    int u = 0;
    int b = matrix.size() - 1;
    int l = 0;
    int r = matrix[0].size() - 1;

    while (true) {
        // 向右移动到最右
        for (int i = l; i <= r; ++i) res.push_back(matrix[u][i]);
        // 更新上边界，若上边界大于下边界，则遍历完成
        if (++u > b) break;

        // 向下移动到最下
        for (int i = u; i <= b; ++i) res.push_back(matrix[i][r]);
        // 更新右边界，若右边界小于左边界，则遍历完成
        if (--r < l) break;

        // 向左移动到最左
        for (int i = r; i >= l; --i) res.push_back(matrix[b][i]);
        // 更新下边界，若下边界小于上边界，则遍历完成
        if (--b < u) break;

        // 向上移动到最上
        for (int i = b; i >= u; --i) res.push_back(matrix[i][l]);
        // 更新左边界，若左边界大于右边界，则遍历完成
        if (++l > r) break;
    }

    return res;
}
```

### 48. 旋转图像

挺有趣的也，先对角翻折，再对称翻折。

```cpp
void rotate(vector<vector<int>>& matrix) {
    int n = matrix.size();

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n/2; ++j) {
            std::swap(matrix[i][j], matrix[i][n-1-j]);
        }
    }
}
```

### 240. 搜索二维矩阵

```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int m = matrix.size();
    int n = matrix[0].size();

    int row = 0;
    int col = n - 1;
    while (row < m && col >= 0) {
        if (matrix[row][col] == target) return true;
        else if (matrix[row][col] < target) ++row;
        else if (matrix[row][col] > target) --col;
    }

    return false;
}
```

## 链表

### 160. 相交链表

```cpp
ListNode *getIntersectionNode(ListNode *ha, ListNode *hb) {
    ListNode* la = ha;
    ListNode* lb = hb;

    while (la != lb) {
        if (la != nullptr) la = la->next;
        else la = hb;

        if (lb != nullptr) lb = lb->next;
        else lb = ha;
    }

    return la;
}
```

### 206. 反转链表

```cpp
ListNode* reverseList(ListNode* head) {
    if (head == nullptr || head->next == nullptr) return head;

    ListNode* tail = head;
    head = reverseList(tail->next);
    tail->next->next = tail;
    tail->next = nullptr;

    return head;
}
```

### 234. 回文链表

注意第 8 行处理了节点个数为奇数的情况。

```cpp {8}
bool isPalindrome(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
    }
    if (fast != nullptr) slow = slow->next;

    fast = reverse(slow);
    slow = head;
    while (fast != nullptr) {
        if (slow->val != fast->val) return false;
        slow = slow->next;
        fast = fast->next;
    }

    return true;
}

ListNode* reverse(ListNode* head) {
    if (head == nullptr || head->next == nullptr) return head;

    ListNode* tail = head;
    head = reverse(tail->next);
    tail->next->next = tail;
    tail->next = nullptr;

    return head;
}
```

### 141. 环形链表

```cpp
bool hasCycle(ListNode *head) {
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
        if (fast == slow) return true;
    }

    return false;
}
```

### 142. 环形链表 II

相遇的时候，让 slow 复位即可，另外此时 fast 不需要走快。

```cpp {9}
ListNode *detectCycle(ListNode *head) {
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            slow = head;
            while (slow != fast) {
                slow = slow->next;
                fast = fast->next;
            }
            return slow;
        }
    }

    return nullptr;
}
```

### 21. 合并两个有序链表

```cpp
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode* dummy = new ListNode(-1, nullptr);
    ListNode* cur = dummy;

    while (l1 != nullptr && l2 != nullptr) {
        ListNode** tmp = (l1->val <= l2->val ? &l1 : &l2);
        cur->next = *tmp;
        *tmp = (*tmp)->next;
        cur = cur->next;
    }
    ListNode** res = (l1 != nullptr ? &l1 : &l2);
    cur->next = *res;

    return dummy->next;
}
```

### 2. 两数相加

腾讯 PCG 技术线-编程系统方向一面手撕算法，ACM 模式。

```cpp
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode* dummy = new ListNode(-1, nullptr);
    ListNode* cur = dummy;

    int carry = 0;
    while (l1 != nullptr && l2 != nullptr) {
        int s = l1->val + l2->val + carry;
        int r = s % 10;
        carry = s / 10;
        cur->next = new ListNode(r, nullptr);
        cur = cur->next;
        l1 = l1->next;
        l2 = l2->next;
    }

    ListNode** res = (l1 != nullptr ? &l1 : &l2);
    while (*res != nullptr) {
        int s = (*res)->val + carry;
        int r = s % 10;
        carry = s / 10;
        cur->next = new ListNode(r, nullptr);
        cur = cur->next;
        *res = (*res)->next;
    }

    if (carry != 0) {
        cur->next = new ListNode(carry, nullptr);
    }

    return dummy->next;
}
```

### 19. 删除链表的倒数第 N 个结点

```cpp
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode* dummy = new ListNode(-1, head);
    ListNode* left = dummy;
    ListNode* right = dummy;

    while (n-- >= 0) {
        right = right->next;
    }

    while (right != nullptr) {
        left = left->next;
        right = right->next;
    }

    left->next = left->next->next;

    return dummy->next;
}
```

### 24. 两两交换链表中的节点

```cpp
ListNode* swapPairs(ListNode* head) {
    if (head == nullptr || head->next == nullptr) return head;

    ListNode* next = head->next;
    head->next = swapPairs(next->next);
    next->next = head;

    return next;
}
```

### 25. K 个一组翻转链表

```cpp
ListNode* reverseKGroup(ListNode* head, int k) {
    ListNode* dummy = new ListNode(0, head);
    ListNode* prev = dummy;
    ListNode* curr = head;
    ListNode* next = nullptr;

    int len = 0;
    while (curr != nullptr) {
        ++len;
        curr = curr->next;
    }

    curr = head;
    for (int i = 0; i < len / k; ++i) {
        for (int j = 0; j < k - 1; ++j) {
            next = curr->next;
            curr->next = next->next;
            next->next = prev->next;
            prev->next = next;
        }
        prev = curr;
        curr = prev->next;
    }

    return dummy->next;
}
```

### 138. 随机链表的复制

我感觉我的解法真不错。

```cpp
Node* copyRandomList(Node* head) {
    if (head == nullptr) return head;

    Node* cur = head;
    Node* tmp = nullptr;
    while (cur != nullptr) {
        tmp = new Node(cur->val);
        tmp->next = cur->next;
        cur->next = tmp;
        cur = cur->next->next;
    }

    cur = head;
    while (cur != nullptr) {
        if (cur->random != nullptr) {
            cur->next->random = cur->random->next;
        }
        cur = cur->next->next;
    }

    Node* res = head->next;
    cur = head;
    tmp = res;
    while (cur != nullptr) {
        cur->next = cur->next->next;
        if (cur->next != nullptr) {
            tmp->next = cur->next->next;
        } else {
            tmp->next = nullptr;
        }
        cur = cur->next;
        tmp = tmp->next;
    }

    return res;
}
```

当然，最简单的还得是记录 old -> new 的映射关系，然后查表复制。

```cpp
Node* copyRandomList(Node* head) {
    unordered_map<Node*, Node*> m;

    Node* cur = head;
    while (cur != nullptr) {
        m[cur] = new Node(cur->val);
        cur = cur->next;
    }

    cur = head;
    while (cur != nullptr) {
        m[cur]->next = m[cur->next];
        m[cur]->random = m[cur->random];
        cur = cur->next;
    }

    return m[head];
}
```

### 148. 排序链表

```cpp
ListNode* sortList(ListNode* head) {
    return sortList(head, nullptr);
}

ListNode* sortList(ListNode*head, ListNode* tail) {
    if (head == nullptr) return head;

    if (head->next == tail) {
        head->next = nullptr;
        return head;
    }

    ListNode* slow = head;
    ListNode* fast = head;
    while (fast != tail) {
        slow = slow->next;
        fast = fast->next;
        if (fast != tail) {
            fast = fast->next;
        }
    }
    ListNode* mid = slow;
    return merge(sortList(head, mid), sortList(mid, tail));
}

ListNode* merge(ListNode* head1, ListNode* head2) {
    ListNode* dummy = new ListNode(-1);
    ListNode* curr = dummy;
    ListNode* l1 = head1;
    ListNode* l2 = head2;

    while (l1 != nullptr && l2 != nullptr) {
        ListNode** tmp = (l1->val <= l2->val ? &l1 : &l2);
        curr->next = *tmp;
        *tmp = (*tmp)->next;
        curr = curr->next;
    }

    ListNode* tmp = (l1 != nullptr ? l1 : l2);
    curr->next = tmp;

    return dummy->next;
}
```

### 23. 合并 K 个升序链表

```cpp
ListNode* mergeKLists(vector<ListNode*>& lists) {
    ListNode* dummy = new ListNode(-1, nullptr);
    ListNode* cur = dummy;

    auto cmp = [](ListNode* a, ListNode* b) -> bool {
        return a->val > b->val;
    };

    std::priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> q(cmp);
    for (const auto& list : lists) {
        if (list != nullptr) q.push(list);
    }

    while (!q.empty()) {
        ListNode* tmp = q.top();
        q.pop();
        cur->next = tmp;
        cur = cur->next;
        if (tmp->next != nullptr) q.push(tmp->next);
    }

    return dummy->next;
}
```

### 146. LRU 缓存

要是遇上寿司 LRU，那就真寄了。

```cpp
class Node {
public:
    Node(int key, int val) : key(key), val(val) { }

    int key;
    int val;
    Node *prev;
    Node *next;
};

class DoubleList {
public:
    DoubleList() {
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head->next = tail;
        tail->prev = head;
        size = 0;
    }

    void add_last(Node* v) {
        v->prev = tail->prev;
        v->prev->next = v;
        v->next = tail;
        tail->prev = v;
        ++size;
    }

    void remove(Node* v) {
        v->prev->next = v->next;
        v->next->prev = v->prev;
        --size;
    }

    Node* remove_first() {
        if (head == nullptr) return nullptr;

        Node* first = head->next;
        remove(first);
        return first;
    }

    int size = 0;
    Node* head = nullptr;
    Node* tail = nullptr;
};

class LRUCache {
public:
    LRUCache(int cap) : cap(cap) {
        cache = new DoubleList();
    }

    int get(int key) {
        if (m.find(key) == m.end()) {
            return -1;
        }
        make_recent(key);
        return m[key]->val;
    }

    void put(int key, int val) {
        if (m.find(key) != m.end()) {
            remove_key(key);
            add_recent(key, val);
            return ;
        }

        if (cap == cache->size) {
            remove_recent();
        }
        add_recent(key, val);
    }

    void make_recent(int key) {
        Node* tmp = m[key];
        cache->remove(tmp);
        cache->add_last(tmp);
    }

    void add_recent(int key, int val) {
        Node* tmp = new Node(key, val);
        cache->add_last(tmp);
        m[key] = tmp;
    }

    void remove_key(int key) {
        Node* tmp = m[key];
        cache->remove(tmp);
        m.erase(key);
    }

    void remove_recent() {
        Node* tmp = cache->remove_first();
        int remove_key = tmp->key;
        m.erase(remove_key);
    }

private:
    int cap;
    std::unordered_map<int, Node*> m;
    DoubleList *cache;
};
```

## 二叉树

### 94. 二叉树的中序遍历

```cpp
vector<int> inorderTraversal(TreeNode* root) {
    vector<int> res;
    stack<TreeNode*> stk;

    while (root != nullptr || !stk.empty()) {
        while (root != nullptr) {
            stk.push(root);
            root = root->left;
        }
        root = stk.top();
        stk.pop();
        res.push_back(root->val);
        root = root->right;
    }

    return res;
}
```

最基础的递归写法。

```cpp
vector<int> inorderTraversal(TreeNode* root) {
    vector<int> res;

    auto inorder = [&res](auto&& self, TreeNode* root) -> void {
        if (root == nullptr) return ;

        self(self, root->left);
        res.push_back(root->val);
        self(self, root->right);
    };
    inorder(inorder, root);

    return res;
}
```

### 104. 二叉树的最大深度

```cpp
int maxDepth(TreeNode* root) {
    if (root == nullptr) return 0;

    return std::max(maxDepth(root->left), maxDepth(root->right)) + 1;
}
```

### 226. 翻转二叉树

```cpp
TreeNode* invertTree(TreeNode* root) {
    if (root == nullptr) return root;

    std::swap(root->left, root->right);
    root->left = invertTree(root->left);
    root->right = invertTree(root->right);

    return root;
}
```

### 101. 对称二叉树

```cpp
bool isSymmetric(TreeNode* root) {
    auto check = [](auto&& self, TreeNode* left, TreeNode* right) -> bool {
        if (left == right) return true;
        else if (left == nullptr || right == nullptr) return false;

        if (left->val != right->val) return false;
        else return self(self, left->left, right->right) && self(self, left->right, right->left);
    };
    return check(check, root->left, root->right);
}
```

### 543. 二叉树的直径

```cpp
int diameterOfBinaryTree(TreeNode* root) {
    int res = 0;

    auto traversal = [&res](auto&& self, TreeNode* root) -> int {
        if (root == nullptr) return 0;

        int left = self(self, root->left);
        int right = self(self, root->right);
        res = std::max(res, left + right);

        return std::max(left, right) + 1;
    };

    traversal(traversal, root);

    return res;
}
```

### 102. 二叉树的层序遍历

```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> res;
    if (root == nullptr) return res;

    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int n = q.size();
        res.push_back(vector<int>());
        for (int i = 0; i < n; ++i) {
            TreeNode* cur = q.front();
            q.pop();
            res.back().push_back(cur->val);
            if (cur->left != nullptr) q.push(cur->left);
            if (cur->right != nullptr) q.push(cur->right);
        }
    }

    return res;
}
```

### 108. 将有序数组转换为二叉搜索树

```cpp
TreeNode* sortedArrayToBST(vector<int>& nums) {
    int n = nums.size();

    auto build = [&nums](auto&& self, int left, int right) -> TreeNode* {
        if (left > right) return nullptr;

        int mid = (left + right) >> 1;
        TreeNode* root = new TreeNode(nums[mid]);
        root->left = self(self, left, mid - 1);
        root->right = self(self, mid + 1, right);

        return root;
    };

    return build(build, 0, n - 1);
}
```

### 98. 验证二叉搜索树

```cpp
bool isValidBST(TreeNode* root) {
    if (root == nullptr) return true;
    long long pre = std::numeric_limits<long long>::min();
    bool res = true;

    auto inorder = [&pre, &res](auto&& self, TreeNode* root) -> void {
        if (root == nullptr) return ;

        self(self, root->left);
        if (pre >= root->val) res = false;
        pre = root->val;
        self(self, root->right);
    };
    inorder(inorder, root);

    return res;
}
```

### 230. 二叉搜索树中第 K 小的元素

```cpp
int kthSmallest(TreeNode* root, int k) {
    int res = 0;

    auto inorder = [&res, &k](auto&& self, TreeNode* root) -> void {
        if (root == nullptr) return ;

        self(self, root->left);
        --k;
        if (k == 0) res = root->val;
        self(self, root->right);
    };
    inorder(inorder, root);

    return res;
}
```

### 199. 二叉树的右视图

```cpp
vector<int> rightSideView(TreeNode* root) {
    vector<int> res;
    if (root == nullptr) return res;

    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int n = q.size();
        for (int i = 0; i < n; ++i) {
            TreeNode* cur = q.front();
            q.pop();

            if (cur->left != nullptr) q.push(cur->left);
            if (cur->right != nullptr) q.push(cur->right);
            if (i == n - 1) res.push_back(cur->val);
        }
    }

    return res;
}
```

### 114. 二叉树展开为链表

```cpp
void flatten(TreeNode* root) {
    if (root == nullptr) return ;

    TreeNode* left = root->left;
    TreeNode* right = root->right;
    flatten(root->left);
    flatten(root->right);

    root->left = nullptr;
    if (left != nullptr) {
        root->right = left;
        TreeNode* temp = left;
        while (temp->right != nullptr) {
            temp = temp->right;
        }
        temp->right = right;
    } else {
        root->right = right;
    }
}
```

### 105. 从前序与中序遍历序列构造二叉树

关键在于计算 `leftsize`。

```cpp {16}
TreeNode* buildTree(vector<int>& pre, vector<int>& ino) {
    int n = pre.size();

    auto build = [&pre, &ino](auto&& self, int pl, int pr, int il, int ir) -> TreeNode* {
        if (pl > pr) return nullptr;

        int rootval = pre[pl];
        int index = 0;
        for (int i = il; i <= ir; ++i) {
            if (ino[i] == rootval) {
                index = i;
                break;
            }
        }

        int leftsize = index - il;
        TreeNode* root = new TreeNode(rootval);
        root->left = self(self, pl+1, pl+1+leftsize-1, il, index-1);
        root->right = self(self, pl+1+leftsize-1+1, pr, index+1, ir);

        return root;
    };

    return build(build, 0, n-1, 0, n-1);
}
```

### 437. 路径总和 III

```cpp
int pathSum(TreeNode* root, int t) {
    int res = 0;
    if (root == nullptr) return res;

    // 以 root 为根的树的路径和为 c
    auto dfs2 = [&res, t](auto&& self, TreeNode* root, long long c) -> void {
        if (root == nullptr) return ;

        c += root->val;
        if (c == t) ++res;
        self(self, root->left, c);
        self(self, root->right, c);
    };

    // 以 root 为根的路径和等于 t 的有多少条
    auto dfs1 = [&dfs2](auto&& self, TreeNode* root) -> void {
        if (root == nullptr) return ;

        dfs2(dfs2, root, 0);
        self(self, root->left);
        self(self, root->right);
    };
    dfs1(dfs1, root);

    return res;
}
```

### 236. 二叉树的最近公共祖先

```cpp
bool find(TreeNode* root, TreeNode* t) {
    if (root == nullptr) return false;
    if (root == t) return true;
    return find(root->left, t) || find(root->right, t);
}

TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (root == nullptr) return nullptr;
    if (root == p || root == q) return root;

    if (find(root->left, p) && find(root->left,q)) {
        return lowestCommonAncestor(root->left, p, q);
    } else if (find(root->right, p) && find(root->right, q)) {
        return lowestCommonAncestor(root->right, p, q);
    }
    return root;
}
```

### 124. 二叉树中的最大路径和

```cpp
int maxPathSum(TreeNode* root) {
    if (root == nullptr) return 0;
    int res = std::numeric_limits<int>::min();

    auto dfs = [&res](auto&& self, TreeNode* root) -> int {
        if (root == nullptr) return 0;

        int left = self(self, root->left);
        int right = self(self, root->right);

        res = std::max(res, left + root->val + right);

        return std::max(std::max(left, right) + root->val, 0);
    };
    dfs(dfs, root);

    return res;
}

```

## 图论

### 200. 岛屿数量

这题数据有点大，BFS 写得随便一点就 TLE 了，��� 如下面这个错误示范：

```cpp
int numIslands(vector<vector<char>>& grid) {
    int m = grid.size();
    int n = grid[0].size();

    vector<array<int, 2>> dirs = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1},
    };

    auto adj = [&grid, &dirs, m, n](int x, int y) -> vector<array<int, 2>> {
        vector<array<int, 2>> res;
        for (auto [dx, dy] : dirs) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || nx >= m || ny < 0 || ny >= n) continue;
            res.push_back({nx, ny});
        }
        return res;
    };

    auto bfs = [&grid, &adj, m, n](int i, int j) -> void {
        queue<array<int, 2>> q;
        q.push({i, j});
        while (!q.empty()) {
            int n = q.size();
            for (int i = 0; i < n; ++i) {
                auto [x, y] = q.front();
                q.pop();
                grid[x][y] = '0';
                for (auto [nx, ny] : adj(x, y)) {
                    if (grid[nx][ny] == '1') {
                        q.push({nx, ny});
                    }
                }
            }
        }
    };

    int res = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] == '1') {
                bfs(i, j);
                ++res;
            }
        }
    }
    return res;
}
```

按下面这样调整染色时机，也即入队的时候立刻染色，防止重复入队。

```diff
diff --git a/a.cpp b/a.cpp
index 3eccba0..91a8eb2 100644
--- a/a.cpp
+++ b/a.cpp
@@ -20,15 +20,16 @@ int numIslands(vector<vector<char>>& grid) {
     auto bfs = [&grid, &adj, m, n](int i, int j) -> void {
         queue<array<int, 2>> q;
         q.push({i, j});
+        grid[i][j] = '0';
         while (!q.empty()) {
             int n = q.size();
             for (int i = 0; i < n; ++i) {
                 auto [x, y] = q.front();
                 q.pop();
-                grid[x][y] = '0';
                 for (auto [nx, ny] : adj(x, y)) {
                     if (grid[nx][ny] == '1') {
                         q.push({nx, ny});
+                        grid[nx][ny] = '0';
                     }
                 }
             }
```

当然，dfs 也是极好的。

```cpp
int numIslands(vector<vector<char>>& grid) {
    int m = grid.size();
    int n = grid[0].size();

    vector<array<int, 2>> dirs = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    };

    auto adj = [&dirs, m, n](int x, int y) -> vector<array<int, 2>> {
        vector<array<int, 2>> res;
        for (const auto [dx, dy] : dirs) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || nx >= m || ny < 0 || ny >= n) continue;
            res.push_back({nx, ny});
        }
        return res;
    };

    auto dfs = [&adj, &grid](auto&& self, int x, int y) -> void {
        if (grid[x][y] == '0') return ;

        grid[x][y] = '0';
        for (const auto [nx, ny] : adj(x, y)) {
            self(self, nx, ny);
        }
    };

    int res = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] == '1') {
                ++res;
                dfs(dfs, i, j);
            }
        }
    }

    return res;
}
```

### 994. 腐烂的橘子

```cpp
int orangesRotting(vector<vector<int>>& grid) {
    int m = grid.size();
    int n = grid[0].size();

    vector<array<int, 2>> dirs = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1},
    };

    auto adj = [&grid, &dirs, m, n](int x, int y) -> vector<array<int, 2>> {
        vector<array<int, 2>> res;
        for (const auto [dx, dy] : dirs) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || nx >= m || ny < 0 || ny >= n) continue;
            res.push_back({nx, ny});
        }
        return res;
    };

    int res = 0;
    int cnt = 0;
    queue<array<int, 2>> q;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] != 0) ++cnt;
            if (grid[i][j] == 2) ++res;
            if (grid[i][j] == 2) q.push({i, j});
        }
    }

    if (res == 0 && cnt != 0) return -1;
    else if (res == cnt) return 0;

    int time = -1;
    while (!q.empty()) {
        int t = q.size();
        for (int i = 0; i < t; ++i) {
            const auto [x, y] = q.front();
            q.pop();
            for (const auto [nx, ny] : adj(x, y)) {
                if (grid[nx][ny] == 1) {
                    q.push({nx, ny});
                    grid[nx][ny] = 0;
                    ++res;
                }
            }
        }
        ++time;
    }

    return res == cnt ? time : -1;
}
```

### 207. 课程表

```cpp
bool canFinish(int n, vector<vector<int>>& pres) {
    vector<vector<int>> g(n, vector<int>());
    vector<int> ind(n, 0);
    for (const auto pre : pres) {
        int u = pre[0];
        int v = pre[1];
        g[v].push_back(u);
        ind[u]++;
    }

    queue<int> q;
    for (int i = 0; i < n; ++i) {
        if (ind[i] == 0) q.push(i);
    }

    int cnt = q.size();
    while (!q.empty()) {
        int t = q.size();
        for (int i = 0; i < t; ++i) {
            int u = q.front();
            q.pop();
            for (const auto v : g[u]) {
                ind[v]--;
                if (ind[v] == 0) {
                    q.push(v);
                    ++cnt;
                }
            }
        }
    }

    return cnt == n;
}
```

### 208. 实现 Trie

```cpp
class Trie {
public:
    Trie() : child(26), is_word(false) { }

    void insert(string word) {
        Trie* cur = this;
        for (char c : word) {
            c -= 'a';
            if (cur->child[c] == nullptr) {
                cur->child[c] = new Trie();
            }
            cur = cur->child[c];
        }
        cur->is_word = true;
    }

    bool search(string word) {
        Trie* res = search_prefix(word);
        return res != nullptr && res->is_word;
    }

    bool startsWith(string prefix) {
        return search_prefix(prefix) != nullptr;
    }

private:
    Trie* search_prefix(string prefix) {
        Trie* cur = this;
        for (char c : prefix) {
            c -= 'a';
            if (cur->child[c] == nullptr) {
                return nullptr;
            }
            cur = cur->child[c];
        }
        return cur;
    }

    vector<Trie*> child;
    bool is_word;
};
```

## 回溯

### 46. 全排列

```cpp
vector<vector<int>> permute(vector<int>& nums) {
    int n = nums.size();
    vector<vector<int>> res;
    if (n == 0) return res;
    vector<int> vis(n, 0);
    vector<int> path;

    auto backtracking = [&nums, &res, &vis, n](auto&& self, vector<int>& path) -> void {
        if (path.size() == n) {
            res.push_back(path);
            return ;
        }


        for (int i = 0; i < n; ++i) {
            if (vis[i] == 1) continue;
            path.push_back(nums[i]);
            vis[i] = 1;
            self(self, path);
            vis[i] = 0;
            path.pop_back();
        }
    };

    backtracking(backtracking, path);

    return res;
}
```

### 78. 子集

全排列和子集是两个很典型的回溯题目，我这里用了两种不同的写法。首先需要明确回溯是递归的写法，那么就需要定义递归边界条件，但代码里看起来都没有写，其实是隐含了。对于全排列问题，由于元素的个数是有限的，那么用 `vis` 数组，跳过已经使用的数字，防止无限递归下去，即使有一些不合理的排列情况，反正只有不添加到最后的结果集里面，也无所谓了。对于子集问题，则使用了 `self(self, path, i+1)` 这里的 `i+1` 以及 `i < n` 共同构成递归的边界条件。

```cpp
vector<vector<int>> subsets(vector<int>& nums) {
    int n = nums.size();
    vector<vector<int>> res;
    vector<int> path;

    auto backtracking = [&res, &nums, n](auto&& self, vector<int>& path, int t) -> void {
        res.push_back(path);

        for (int i = t; i < n; ++i) {
            path.push_back(nums[i]);
            self(self, path, i+1);
            path.pop_back();
        }
    };
    backtracking(backtracking, path, 0);

    return res;
}
```

### 17. 电话号码的字母组合

```cpp
vector<string> letterCombinations(string digits) {
    std::unordered_map<char, string> m {
        {'1', ""},     {'2', "abc"}, {'3', "def"},
        {'4', "ghi"},  {'5', "jkl"}, {'6', "mno"},
        {'7', "pqrs"}, {'8', "tuv"}, {'9', "wxyz"},
    };

    vector<string> res;
    string path { "" };
    int n = digits.size();
    if (n == 0) return res;

    auto backtracking = [&digits, &res, &path, &m, n](auto&& self, int k) -> void {
        if (path.size() == n) {
            res.push_back(path);
            return ;
        }

        for (int i = k; i < n; ++i) {
            for (auto& c : m[digits[i]]) {
                path.push_back(c);
                self(self, i+1);
                path.pop_back();
            }
        }
    };
    backtracking(backtracking, 0);

    return res;
}
```

### 39. 组合总和

```cpp
vector<vector<int>> combinationSum(vector<int>& candidates, int t) {
    int n = candidates.size();
    vector<vector<int>> res;
    vector<int> path;

    auto backtracking = [&res, &candidates, &path, n, t](auto&& self, int s, int k) -> void {
        if (s > t) return ;
        if (s == t) {
            res.push_back(path);
            return ;
        }

        for (int i = k; i < n; ++i) {
            path.push_back(candidates[i]);
            s += candidates[i];
            self(self, s, i);
            s -= candidates[i];
            path.pop_back();
        }
    };
    backtracking(backtracking, 0, 0);

    return res;
}
```

### 22. 括号生成

这里比较难处理的是如何保证生成的括号对是有效的，我这里的处理思路是增加一个变量 `c` 用来记录左右括号的平衡关系，可以理解为表示有多少个合法的左括号。那么，当 `c < 0` 的时候，显然得终止掉，因为不可能再使其合法了。同样的，当 `path.size() == 0` 的时候，也要用 `c == 0` 来判断是否合法。

```cpp {7,9}
vector<string> generateParenthesis(int n) {
    vector<string> res;
    string path;
    n *= 2;

    auto backtracking = [&res, &path, n](auto&& self, int k, int c) -> void {
        if (c < 0) return ;

        if (path.size() == n && c == 0) {
            res.push_back(path);
            return ;
        }

        for (int i = k; i < n; ++i) {
            ++c;
            path.push_back('(');
            self(self, i+1, c);
            path.pop_back();
            --c;

            --c;
            path.push_back(')');
            self(self, i+1, c);
            path.pop_back();
            ++c;
        }
    };

    backtracking(backtracking, 0, 0);

    return res;
}
```

### 79. 单词搜索

```cpp
bool exist(vector<vector<char>>& board, string word) {
    int m = board.size();
    int n = board[0].size();
    if (m == 0) return false;
    vector<vector<int>> vis(m, vector<int>(n, 0));

    auto backtracking = [&board, &word, &vis, m, n](auto&& self, int i, int j, int idx) -> bool {
        if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != word[idx] || vis[i][j]) {
            return false;
        }

        if (idx == word.size() - 1) return true;
        vis[i][j] = true;
        bool res = self(self, i + 1, j, idx + 1) ||
                   self(self, i - 1, j, idx + 1) ||
                   self(self, i, j + 1, idx + 1) ||
                   self(self, i, j - 1, idx + 1);
        vis[i][j] = false;

        return res;
    };

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (backtracking(backtracking, i, j, 0)) {
                return true;
            }
        }
    }

    return false;
}
```

### 131. 分割回文串

```cpp
vector<vector<string>> partition(string s) {
    auto check = [&s](int l, int r) -> bool {
        while (l < r) {
            if (s[l] != s[r]) return false;
            ++l;
            --r;
        }
        return true;
    };

    int n = s.size();
    vector<vector<string>> res;
    vector<string> path;

    auto dfs = [&res, &path, &check, &s, n](auto&& self, int i) {
        if (i == n) {
            res.push_back(path);
            return ;
        }

        for (int j = i; j < n; ++j) {
            if (check(i, j)) {
                path.push_back(s.substr(i, j-i+1));
                self(self, j + 1);
                path.pop_back();
            }
        }
    };
    dfs(dfs, 0);

    return res;
}
```

### 51. N 皇后

```cpp
vector<vector<string>> solveNQueens(int n) {
    vector<vector<string>> res;
    vector<string> track(n, string(n, '.'));
    if (n == 0) return res;

    auto check = [&track, n](int row, int col) -> bool {
        // 依次检查列、主对角线、行对角线
        for (int i = 0; i < row; ++i) {
            if (track[i][col] == 'Q') return false;
        }

        for (int i = row-1, j = col-1; i >= 0 && j >= 0; --i, --j) {
            if (track[i][j] == 'Q') return false;
        }

        for (int i = row-1, j = col+1; i >= 0 && j <= n; --i, ++j) {
            if (track[i][j] == 'Q') return false;
        }

        return true;
    };

    auto backtracking = [&res, &track, &check, n](auto&& self, int row) -> void {
        if (row == n) {
            res.push_back(track);
            return ;
        }

        for (int col = 0; col < n; ++col) {
            if (check(row, col)) {
                track[row][col] = 'Q';
                self(self, row + 1);
                track[row][col] = '.';
            }
        }
    };
    backtracking(backtracking, 0);

    return res;
}
```

## 二分查找

### 35. 搜索插入位置

搜索左右边界都是可以的，但一定注意如果可能不存在需要做特殊处理。

```cpp
// [l, m], (m, r]
int searchInsert(vector<int>& nums, int t) {
    int n = nums.size();
    int l = 0;
    int r = n - 1;
    while (l < r) {
        int m = (l + r + 1) >> 1;
        if (nums[m] <= t) l = m;
        else r = m - 1;
    }

    if (nums[l] < t) ++l;
    return l;
}
```

```cpp
// [l, m), [m, r]
int searchInsert(vector<int>& nums, int t) {
    int n = nums.size();
    int l = 0;
    int r = n - 1;
    while (l < r) {
        int m = (l + r) >> 1;
        if (nums[m] >= t) r = m;
        else l = m + 1;
    }

    if (nums[l] < t) ++l;
    return l;
}
```

### 74. 搜索二维矩阵

```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int m = matrix.size();
    int n = matrix[0].size();

    int row = 0;
    int col = n - 1;

    while (row < m && col >= 0) {
        if (matrix[row][col] > target) {
            --col;
        } else if (matrix[row][col] == target) {
            return true;
        } else if (matrix[row][col] < target) {
            ++row;
        }
    }

    return false;
}
```

### 34. 在排序数组中查找元素的第一个和最后一个位置

```cpp
vector<int> searchRange(vector<int>& nums, int t) {
    int n = nums.size();
    if (n == 0) return {-1, -1};

    // 找区间的左边界
    int ll = 0, lr = n - 1;
    while (ll < lr) {
        int m = (ll + lr) >> 1;
        if (nums[m] >= t) lr = m;
        else ll = m + 1;
    }

    // 找区间的右边界
    int rl = 0, rr = n - 1;
    while (rl < rr) {
        int m = (rl + rr + 1) >> 1;
        if (nums[m] <= t) rl = m;
        else rr = m - 1;
    }
    if (nums[ll] != t) return {-1, -1};

    return {ll, rl};
}
```

### 33. 搜索旋转排序数组

```cpp
int search(vector<int>& nums, int target) {
    int n = nums.size();
    if (n == 0) return -1;
    if (n == 1) return nums[0] == target ? 0 : -1;

    int l = 0;
    int r = n - 1;
    while (l <= r) {
        int m = (l + r) >> 1;
        if (nums[m] == target) return m;
        if (nums[0] <= nums[m]) {
            if (nums[0] <= target && target < nums[m]) {
                r = m - 1;
            } else {
                l = m + 1;
            }
        } else {
            if (nums[m] < target && target <= nums[n-1]) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
    }

    return -1;
}
```

### 153. 寻找旋转排序数组中的最小值

唯一不同的只有第 7 行的判断条件。

```cpp {7}
int findMin(vector<int>& nums) {
    int n = nums.size();
    int l = 0;
    int r = n - 1;
    while (l < r) {
        int m = (l + r) >> 1;
        if (nums[m] <= nums[r]) {
            r = m;
        } else {
            l = m + 1;
        }
    }
    return nums[l];
}
```

### 4. 寻找两个正序数组的中位数

```cpp
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    priority_queue<int, vector<int>, std::greater<int>> large;
    priority_queue<int, vector<int>, std::less<int>> small;

    auto add_num = [&large, &small](int val) -> void {
        if (large.size() > small.size()) {
            large.push(val);
            small.push(large.top());
            large.pop();
        } else {
            small.push(val);
            large.push(small.top());
            small.pop();
        }
    };

    for (const auto& num : nums1) add_num(num);
    for (const auto& num : nums2) add_num(num);

    if (large.size() > small.size()) {
        return large.top();
    } else if (large.size() == small.size()) {
        return (large.top() + small.top()) / 2.0;
    } else {
        return small.top();
    }
}
```

## 栈

单调栈模板

```cpp
stack<int> stk;
for(int i = 0; i < n; i++) {
	while(!stk.empty() && stk.top() > nums[i]) {
		stk.pop();
	}
	stk.push(nums[i]);
}
```

### 20. 有效的括号

```cpp
bool isValid(string s) {
    stack<char> stk;

    auto pair = [](char c) -> char {
        if (c == ')') return '(';
        else if (c == ']') return '[';
        else return '{';
    };

    for (const auto c : s) {
        if (c == '(' || c == '[' || c == '{') {
            stk.push(c);
        } else {
            char u = pair(c);
            if (stk.empty() || stk.top() != u) {
                return false;
            } else {
                stk.pop();
            }
        }
    }

    return stk.empty();
}
```

### 155. 最小栈

```cpp
class MinStack {
public:
    MinStack() {
        head = nullptr;
    }

    void push(int val) {
        if (head == nullptr) {
            head = new Node(val, val);
        } else {
            int min = std::min(val, head->min);
            head = new Node(val, min, head);
        }
    }

    void pop() {
        head = head->next;
    }

    int top() {
        return head->val;
    }

    int getMin() {
        return head->min;
    }

private:
    // 妙啊，在每个节点里保存当前的最小值
    // 剩下的就只要实现一个链栈即可
    struct Node {
        int val;
        int min;
        Node* next;

        Node(int val, int min) : val(val), min(min), next(nullptr) { }
        Node(int val, int min, Node* next) : val(val), min(min), next(next) { }
    };

    Node* head;
};
```

### 394. 字符串解码

不太好想的迭代解法。

```cpp
string decodeString(string s) {
    int n = s.size();
    int num = 0;
    string res { "" };

    std::stack<std::pair<int, string>> stk;
    for (const auto c : s) {
        if (c >= '0' && c <= '9') {
            num = num * 10 + (c - '0');
        } else if (c == '[') {
            stk.push({num, res});
            num = 0;
            res = "";
        } else if (c == ']') {
            auto [tmpn, tmps] = stk.top();
            stk.pop();
            for (int i = 0; i < tmpn; ++i) {
                tmps = tmps + res;
            }
            res = tmps;
        } else {
            res += c;
        }
    }

    return res;
}
```

不太好写的递归解法。

```cpp
string decodeString(string s) {
    int n = s.size();

    auto decode = [&s, n](auto&& self, int& k) -> string {
        int num = 0;
        string res { "" };

        while (k < n) {
            char c = s[k++];
            if (c >= '0' && c <= '9') {
                num = num * 10 + c - '0';
            } else if (c == '[') {
                string tmp = self(self, k);
                for (int i = 0; i < num; ++i) {
                    res += tmp;
                }
                num = 0;
            } else if (c == ']') {
                break;
            } else {
                res += c;
            }
        }
        return res;
    };

    int k = 0;
    string res = decode(decode, k);

    return res;
}
```

### 739. 每日温度

```cpp
vector<int> dailyTemperatures(vector<int>& temps) {
    int n = temps.size();
    vector<int> res(n, 0);
    stack<int> stk;

    for (int i = n - 1; i >= 0; --i) {
        while (!stk.empty() && temps[stk.top()] <= temps[i]) {
            stk.pop();
        }
        res[i] = stk.empty() ? 0 : stk.top() - i;
        stk.push(i);
    }

    return res;
}

```

### 84. 柱状图中最大的矩形

当前高度对应的最大矩形面积的宽度是两边第一个比其高度矮的柱子的下标之差，那么遍历每个高度，求出该高度对应的最大宽度，那么就是该高度对应的最大面积。

```cpp
int largestRectangleArea(vector<int>& h) {
    int n = h.size();
    if (n == 1) return h[0];

    int res = 0;
    stack<int> stk;
    for (int i = 0; i < n; ++i) {
        while (!stk.empty() && h[stk.top()] > h[i]) {
            int len = h[stk.top()];
            stk.pop();
            int weight = i;
            if (!stk.empty()) {
                weight = i - stk.top() - 1;
            }
            res = std::max(res, len * weight);
        }
        stk.push(i);
    }

    while (!stk.empty()) {
        int len = h[stk.top()];
        stk.pop();
        int weight = n;
        if (!stk.empty()) {
            weight = n - stk.top() - 1;
        }
        res = std::max(res, len * weight);
    }

    return res;
}
```

加入哨兵，简化栈是否为空的判断。

```cpp
int largestRectangleArea(vector<int>& h) {
    int res = 0;
    stack<int> stk;
    h.insert(h.begin(), 0);
    h.push_back(0);
    int n = h.size();

    for (int i = 0; i < n; ++i) {
        while (!stk.empty() && h[stk.top()] > h[i]) {
            int cur = stk.top();
            stk.pop();
            int left = stk.top() + 1;
            int right = i - 1;
            res = std::max(res, (right - left + 1) * h[cur]);
        }
        stk.push(i);
    }

    return res;
}
```

## 堆

### 215. 数组中的第 K 个最大元素

默写快排板子罢了。

```cpp
int findKthLargest(vector<int>& nums, int k) {
    int n = nums.size();

    auto qs = [&nums](auto&& self, int l, int r) -> void {
        if (l >= r) return ;

        int x = nums[(l+r)>>1];
        int i = l - 1;
        int j = r + 1;
        while (i < j) {
            do ++i; while (nums[i] < x);
            do --j; while (nums[j] > x);
            if (i < j) std::swap(nums[i], nums[j]);
        }

        self(self, l, j);
        self(self, j+1, r);
    };

    qs(qs, 0, n-1);

    for (const auto& num : nums) {
        n--;
        if (n < k) return num;
    }

    return -1;
}
```

### 347. 前 K 个高频元素

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    int n = nums.size();
    std::unordered_map<int, int> m;
    for (const auto num : nums) m[num]++;
    vector<array<int, 2>> tmp;
    for (const auto [k, v] : m) {
        tmp.push_back({k, v});
    }
    std::sort(tmp.begin(), tmp.end(), [](const array<int, 2>& a, const array<int, 2>& b) -> bool {
        return a[1] > b[1];
    });

    vector<int> res;
    for (int i = 0; i < k; ++i) {
        res.push_back(tmp[i][0]);
    }
    return res;
}
```

### 295. 数据流的中位数

```cpp
class MedianFinder {
public:
    MedianFinder() = default;

    void addNum(int num) {
        if (large.size() > small.size()) {
            large.push(num);
            small.push(large.top());
            large.pop();
        } else {
            small.push(num);
            large.push(small.top());
            small.pop();
        }
    }

    double findMedian() {
        if (large.size() > small.size()) {
            return large.top();
        } else if (large.size() == small.size()) {
            return (large.top() + small.top()) / 2.0;
        } else {
            return small.top();
        }
    }

private:
    std::priority_queue<int, vector<int>, std::greater<int>> large;
    std::priority_queue<int, vector<int>, std::less<int>> small;
};
```

## 贪心算法

### 121. 买卖股票的最佳时机

```cpp
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    vector<int> f(n, 0);

    int min = prices[0];
    for (int i = 1; i < n; ++i) {
        min = std::min(min, prices[i]);
        f[i] = std::max(f[i-1], prices[i] - min);
    }

    return f[n-1];
}
```

### 55. 跳跃游戏

```cpp
bool canJump(vector<int>& nums) {
    int n = nums.size();
    int dis = 0;

    for (int i = 0; i < n; ++i) {
        if (i > dis) return false;
        dis = std::max(dis, i + nums[i]);
    }

    return true;
}
```

### 45. 跳跃游戏 II

```cpp
int jump(vector<int>& nums) {
    int n = nums.size();
    vector<int> f(n, n);
    f[0] = 0;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (j + nums[j] >= i) {
                    f[i] = std::min(f[i], f[j]+1);
                }
            }
        }

    return f[n-1];

}
```

### 763. 划分字母区间

把每个字母在字符串中第一次和最后一次出现的位置记录下来，然后就变成合并区间模板题了。

```cpp
vector<int> partitionLabels(string s) {
    int n = s.size();
    vector<array<int, 2>> m(26, array<int, 2>({n+1, -1}));

    for (int i = 0; i < n; ++i) {
        int c = s[i] - 'a';
        m[c][0] = std::min(m[c][0], i);
        m[c][1] = std::max(m[c][1], i);
    }

    m.erase(std::remove_if(m.begin(), m.end(), [n](array<int, 2>& a) -> bool {
        return a[0] == n + 1;
        }), m.end());
    std::sort(m.begin(), m.end(), [](array<int, 2>& a, array<int, 2>& b) -> bool {
        return a[0] < b[0];
    });

    vector<int> res;

    int t = m.size();
    for (int i = 0, j = 0; i < t; i = j) {
        int l = m[i][0];
        int r = m[i][1];
        if (i == t - 1) res.push_back(r - l + 1);

        for (j = i + 1; j < t; ++j) {
            int u = m[j][0];
            int v = m[j][1];
            if (l <= u && u <= r) {
                l = std::min(l, u);
                r = std::max(r, v);
                if (j == t - 1) res.push_back(r - l + 1);
            } else {
                res.push_back(r - l + 1);
                break;
            }
        }
    }

    return res;
}
```

## 动态规划

### 70. 爬楼梯

```cpp
int climbStairs(int n) {
    if (n == 1) return 1;
    if (n == 2) return 2;

    vector<int> f(n, 0);
    f[0] = 1;
    f[1] = 2;
    for (int i = 2; i < n; ++i) {
        f[i] = f[i-1] + f[i-2];
    }
    return f[n-1];
}
```

### 118. 杨辉三角

```cpp
vector<vector<int>> generate(int n) {
    vector<vector<int>> res;
    res.push_back(vector<int>({1}));
    if (n == 1) return res;
    res.push_back({1, 1});
    if (n == 2) return res;

    for (int i = 2; i < n; ++i) {
        res.push_back(vector<int>());
        res.back().push_back(1);
        vector<int>& pre = res[i-1];
        for (int j = 1; j < pre.size(); ++j) {
            res.back().push_back(pre[j] + pre[j-1]);
        }
        res.back().push_back(1);
    }

    return res;
}
```

### 198. 打家劫舍

```cpp
int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 1) return nums[0];

    vector<int> f(n, 0);
    f[0] = nums[0];
    f[1] = std::max(nums[0], nums[1]);
    for (int i = 2; i < n; ++i) {
        f[i] = std::max(f[i-1], f[i-2] + nums[i]);
    }

    return f[n-1];
}
```

### 279. 完全平方数

```cpp
int numSquares(int n) {
    vector<int> f(n+1, std::numeric_limits<int>::max());
    f[0] = 0;

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j * j <= i; ++j) {
            f[i] = std::min(f[i], f[i-j*j] + 1);
        }
    }

    return f[n];
}
```

### 322. 零钱兑换

```cpp
int coinChange(vector<int>& coins, int n) {
    vector<int> f(n+1, n+1);
    f[0] = 0;
    for (int i = 1; i <= n; ++i) {
        for (const auto coin : coins) {
            if (i - coin >= 0) {
                f[i] = std::min(f[i], f[i-coin] + 1);
            }
        }
    }

    return f[n] != n +1 ? f[n] : -1;
}
```

### 139. 单词拆分

```cpp
bool wordBreak(string s, vector<string>& dict) {
    int n = s.size();

    vector<int> f(n+1, 0);
    f[0] = 1;

    for (int i = 1; i <= n; ++i) {
        for (const auto& word : dict) {
            int m = word.size();
            if (i >= m && s.substr(i-m, m) == word) {
                f[i] |= f[i-m];
            }
        }
    }

    return f[n];
}
```

### 300. 最长递增子序列

```cpp
int lengthOfLIS(vector<int>& nums) {
    int n = nums.size();
    vector<int> f(n, 1);

    int res = 1;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (nums[i] > nums[j]) {
                f[i] = std::max(f[i], f[j] + 1);
                res = std::max(res, f[i]);
            }
        }
    }

    return res;
}
```

### 152. 乘积最大子数组

略微有点意思的题，乘积不像累加那么简单，最大的一定是较大的数相加，乘积里最大的可能是两个较小的负数相乘，因而这里维护两个数组，分别处理正数和负数的情况（可以这么理解）。同时，也因为有负数的出现，`res` 不能初始化为 `0`，最稳妥的 ���� ���� 是初始化成 `nums[]` 中的元素 ��

```cpp {8,10-11}
int maxProduct(vector<int>& nums) {
    int n = nums.size();
    vector<int> f1(n, 1);
    vector<int> f2(n, 1);

    f1[0] = nums[0];
    f2[0] = nums[0];
    int res = nums[0];
    for (int i = 1; i < n; ++i) {
        f1[i] = std::max(std::max(f1[i-1] * nums[i], f2[i-1] * nums[i]), nums[i]);
        f2[i] = std::min(std::min(f2[i-1] * nums[i], f1[i-1] * nums[i]), nums[i]);
        res = std::max(res, std::max(f1[i], f2[i]));
    }
    return res;
}
```

### 416. 分割等和子集

如果是求排列数，外层遍历背包，内层遍历物品。
如果是求组合数，外层遍历物品，内层遍历背包；
没要求就随便写，但一般外层遍历物品写起来舒服一点。

外层遍历物品。

```cpp
bool canPartition(vector<int>& nums) {
    int n = nums.size();
    int s = 0;
    for (int i = 0; i < n; ++i) s += nums[i];
    if (s & 1) return false;

    int w = s >> 1;
    vector<vector<int>> f(n+1, vector<int>(w+1, 0));
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= w; ++j) {
            if (j - nums[i-1] < 0) {
                f[i][j] = f[i-1][j];
            } else {
                f[i][j] = std::max(f[i-1][j], f[i-1][j-nums[i-1]] + nums[i-1]);
            }
        }
    }

    return f[n][w] == w;
}
```

外层遍历背包。

```cpp
bool canPartition(vector<int>& nums) {
    int n = nums.size();
    int s = 0;
    for (int i = 0; i < n; ++i) s += nums[i];
    if (s & 1) return false;

    int w = s >> 1;
    vector<vector<int>> f(w+1, vector<int>(n+1, 0));
    for (int i = 1; i <= w; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (i - nums[j-1] < 0) {
                f[i][j] = f[i][j-1];
            } else {
                f[i][j] = std::max(f[i][j-1], f[i-nums[j-1]][j-1] + nums[j-1]);
            }
        }
    }

    return f[w][n] == w;
}
```

### 32. 最长有效括号

```cpp
int longestValidParentheses(string s) {
    int n = s.size();

    stack<int> stk;
    vector<int> mark(n, 0);

    for (int i = 0; i < n; ++i) {
        if (s[i] == '(') stk.push(i);
        else if (!stk.empty()) stk.pop();
        else mark[i] = 1;
    }

    while (!stk.empty()) {
        mark[stk.top()] = 1;
        stk.pop();
    }

    int res = 0;
    int len = 0;

    for (int i = 0; i < n; ++i) {
        if (mark[i]) {
            len = 0;
            continue;
        }
        ++len;
        res = std::max(res, len);
    }

    return res;
}
```

## 多维动态规划

### 62. 不同路径

```cpp
int uniquePaths(int m, int n) {
    vector<vector<int>> f(m, vector<int>(n, 0));
    for (int i = 0; i < m; ++i) f[i][0] = 1;
    for (int i = 0; i < n; ++i) f[0][i] = 1;

    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            f[i][j] = f[i-1][j] + f[i][j-1];
        }
    }

    return f[m-1][n-1];
}
```

```cpp
int uniquePaths(int m, int n) {
    vector<int> f(n, 1);

    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            f[j] = f[j] + f[j-1];
        }
    }

    return f[n-1];
}
```

### 64. 最 ���� 路径和

```cpp
int minPathSum(vector<vector<int>>& grid) {
    int m = grid.size();
    int n = grid[0].size();

    vector<vector<int>> f(m, vector<int>(n, 0));
    f[0][0] = grid[0][0];
    for (int i = 1; i < m; ++i) f[i][0] = f[i-1][0] + grid[i][0];
    for (int j = 1; j < n; ++j) f[0][j] = f[0][j-1] + grid[0][j];

    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            f[i][j] = std::min(f[i-1][j], f[i][j-1]) + grid[i][j];
        }
    }

    return f[m-1][n-1];
}
```

状压的时候别忘记初始化了。

```cpp {10}
int minPathSum(vector<vector<int>>& grid) {
    int m = grid.size();
    int n = grid[0].size();

    vector<int> f(n, 0);
    f[0] = grid[0][0];
    for (int i = 1; i < n; ++i) f[i] = f[i-1] + grid[0][i];

    for (int i = 1; i < m; ++i) {
        f[0] += grid[i][0];
        for (int j = 1; j < n; ++j) {
            f[j] = std::min(f[j], f[j-1]) + grid[i][j];
        }
    }

    return f[n-1];
}
```

### 5. 最长回文子串

不会动态规划做，我好菜。

```cpp
string longestPalindrome(string s) {
    int n = s.size();
    string res {""};

    auto palindrome = [&res, n](string& s, int l, int r) -> void {
        while (l >= 0 && r < n && s[l] == s[r]) {
            --l;
            ++r;
        }
        string t = s.substr(l+1, r-l-1);
        if (t.size() > res.size()) res = t;
    };

    for (int i = 0; i < n; ++i) {
        palindrome(s, i, i);
        palindrome(s, i, i+1);
    }

    return res;
}
```

### 1143. 最长公共子序列

```cpp
int longestCommonSubsequence(string s1, string s2) {
    int m = s1.size();
    int n = s2.size();

    int res = 0;
    vector<vector<int>> f(m+1, vector<int>(n+1, 0));
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (s1[i-1] == s2[j-1]) {
                f[i][j] = f[i-1][j-1] + 1;
            } else {
                f[i][j] = std::max(f[i][j-1], f[i-1][j]);
            }
        }
    }

    return f[m][n];
}
```

### 72. 编辑距离

```cpp
int minDistance(string s1, string s2) {
    int m = s1.size();
    int n = s2.size();

    vector<vector<int>> f(m+1, vector<int>(n+1, 0));
    for (int i = 0; i <= m; ++i) f[i][0] = i;
    for (int i = 0; i <= n; ++i) f[0][i] = i;

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (s1[i-1] == s2[j-1]) {
                f[i][j] = f[i-1][j-1];
            } else {
                int a = f[i-1][j];
                int b = f[i][j-1];
                int c = f[i-1][j-1];
                f[i][j] = std::min(a, std::min(b, c)) + 1;
            }
        }
    }

    return f[m][n];
}
```

## 技巧

### 136. 只出现一次的数字

```cpp
int singleNumber(vector<int>& nums) {
    int res = 0;

    for (const auto num : nums) res ^= num;

    return res;
}
```

### 169. 多数元素

```cpp
int majorityElement(vector<int>& nums) {
    int n = nums.size();
    int num = nums[0];
    int cnt = 0;

    for (int i = 0; i < n; ++i) {
        if (nums[i] == num) ++cnt;
        else {
            if (cnt > 1) --cnt;
            else {
                num = nums[i];
                cnt = 1;
            }
        }
    }

    return num;
}
```

### 75. 颜色分类

```cpp
void sortColors(vector<int>& nums) {
    int n = nums.size();
    int l = 0;
    int r = n - 1;
    int c = 0;

    while (c <= r) {
        if (nums[c] == 0) {
            std::swap(nums[c], nums[l]);
            ++l;
            ++c;
        } else if (nums[c] == 1) {
            ++c;
        } else if (nums[c] == 2) {
            std::swap(nums[c], nums[r]);
            --r;
        }
    }
}
```

### 31. 下一个排列

```cpp
void nextPermutation(vector<int>& nums) {
    int n = nums.size();

    int l = 0;
    int r = 0;

    for (int i = n - 2; i >= 0; --i) {
        // 从后往前找到第一个逆序的数字
        if (nums[i] < nums[i+1]) {
            l = i;
            // 从后往前找到第一个比 nums[l] 大的数字交换位置
            for (int j = n - 1; j > l; --j) {
                if (nums[j] > nums[l]) {
                    r = j;
                    break;
                }
            }
            std::swap(nums[l], nums[r]);
            // 当然，需要将知乎的数字重新排序
            std::sort(nums.begin()+l+1, nums.end(), std::less<int>());
            return ;
        }
    }

    // 如果一个逆序对都找不到的话，说明已经是最大的排列了，直接将数组 reverse 即可
    std::reverse(nums.begin(), nums.end());
}
```

### 287. 寻找重复数

妙啊。

```cpp
int findDuplicate(vector<int>& nums) {
    int slow = 0;
    int fast = 0;

    do {
        slow = nums[slow];
        fast = nums[nums[fast]];
    } while (slow != fast);

    slow = 0;
    while (slow != fast) {
        slow = nums[slow];
        fast = nums[fast];
    }

    return slow;
}
```
