import LinkedList

class NumArray(object):
    #307. Range Sum Query - Mutable
    def __init__(self, nums):
        self.nums = nums
        
    def update(self, i, val):
        self.nums[i] = val
        
    def sumRange(self, i, j):
        return sum(self.nums[i:j+1])


class Solution(object):

    def get_url_desc(self,func):
        d ={
            self.letterCombinations : ('https://leetcode.com/problems/letter-combinations-of-a-phone-number/#/description'
                                      , 'Given a digit string, return all possible letter combinations that the number could represent.')
            , self.addTwoNumbers : ('https://leetcode.com/problems/add-two-numbers/#/description'
                                    ,'You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit.\nAdd the two numbers and return it as a linked list.\nYou may assume the two numbers do not contain any leading zero, except the number 0 itself.')
            , self.isPalindrome : ('https://leetcode.com/problems/palindrome-number/#/description'
                                   , 'Determine whether an integer is a palindrome. Do this without extra space.')
            , self.romanToInt : ('https://leetcode.com/problems/roman-to-integer/#/description'
            , 'Given a roman numeral, convert it to an integer.\nInput is guaranteed to be within the range from 1 to 3999.')
            , self.setZeroes : ('https://leetcode.com/problems/set-matrix-zeroes/#/description'
                               , 'Given a m x n matrix, if an element is 0,\nset its entire row and column to 0. Do it in place.')
            , self.isScramble : ('https://leetcode.com/problems/scramble-string/#/description'
                                 , 'Given a string s1, we may represent it as a binary tree by partitioning it to two non-empty substrings recursively.\nTo scramble the string, we may choose any non-leaf node and swap its two children.\nFor example, if we choose the node "gr" and swap its two children, it produces a scrambled string "rgeat"')
            , self.getHint : ('https://leetcode.com/problems/bulls-and-cows/#/description'
                       , 'You are playing the following Bulls and Cows game with your friend: You write down a number and ask your friend to guess what the number is. Each time your friend makes a guess, you provide a hint that indicates how many digits in said guess match your secret number exactly in both digit and position (called "bulls") and how many digits match the secret number but locate in the wrong position (called "cows"). Your friend will use successive guesses and hints to eventually derive the secret number.')
            , self.twoSum : ('https://leetcode.com/problems/two-sum/#/description'
                           , 'Given an array of integers, return indices of the two numbers such that they add up to a specific target.You may assume that each input would have exactly one solution, and you may not use the same element twice.')
            , self.threeSum : ('https://leetcode.com/problems/3sum/#/description'
                              , 'Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.')
            , self.getMinimumDifference : ('https://leetcode.com/problems/minimum-absolute-difference-in-bst/#/description'
                 ,'Given a binary search tree with non-negative values, find the minimum absolute difference between values of any two nodes.')
            }

        return d[func]

    def test_function(self,func):
        if func == self.RemoveDupes:
            pass
        
    def twoSum(self,nums = [2, 7, 11, 15], target = 9):
        #1. Two Sum
        from collections import defaultdict
        d = defaultdict(int)
        indexes = {}
        for i,l in enumerate(nums):
            v = target - l
            if d[v]>0:
                return [indexes[v],i]
            d[l] = 1
            indexes[l]=i

    def addTwoNumbers(self, l1, l2):
        #2. Add Two Numbers
        carry = 0
        l=[]
        while l1 or l2 or carry:
            a = 0
            b = 0
            if l1:
                a = l1.val
                l1 = l1.next
            if l2:
                b = l2.val
                l2 = l2.next
            carry, v = divmod(a+b+carry, 10)
            l.append(v)
        return l
    
    def isPalindrome(self, x):
        #9. Palindrome Number
        if x<0:
            return False
        x = list(str(x))
        l = 0
        r = len(x)-1
        while l<r:
            if x[l]!=x[r]:
                return False
            l = l + 1
            r = r - 1
        return True
        
    def romanToInt(self, s):
        #13. Roman to Integer
        romans = {'M': 1000,'D': 500 ,'C': 100,'L': 50,'X': 10,'V': 5,'I': 1}
        count = 0
        l=0
        r = len(s)-1
        v1 = 0
        while l <= r:
            v1 = romans[s[l]]
            if l!=r and v1 <  romans[s[l+1]]:
                count = count +  romans[s[l+1]] - v1
                l = l + 1
            else:
                count = count + v1
            l = l + 1
        return count

    def threeSum(self, nums=[-1, 0, 1, 2, -1, -4]):
        #15. 3Sum
        s = set()
        nums.sort()
        l = []
        for i in xrange(len(nums)-2):
            j = i + 1
            k = len(nums)-1
            while j<k:
                v = nums[i] + nums[j] + nums[k]
                if v == 0 and (nums[i] , nums[j] , nums[k]) not in s:
                    l.append([nums[i] , nums[j] , nums[k]])
                    s.add((nums[i] , nums[j] , nums[k]))
                    j+=1
                elif v < 0:
                    j+=1
                elif v > 0:
                    k+=-1
                else:
                    if nums[i]==nums[k]:
                        break
                    j+=1
        return l
    
    def letterCombinations(self, digits):
        #17. Letter Combinations of a Phone Number
        if len(digits)==0:
            return []
        d = {'0':[' '], '1':[], '2':['a','b','c'], '3':['d','e','f'], '4':['g','h','i'], '5':['j','k','l']
             , '6':['m','n','o'], '7':['p','q','r','s'], '8':['t','u','v'], '9':['w','x','y','z']}
        s = set()
        for l2 in d[digits[0]]:
            s.add(l2)
        for i in xrange(1, len(digits)):
            for l1 in s.copy():
                for l2 in d[digits[i]]:
                    l2 = l1 + l2
                    s.add(l2)
                s.remove(l1)
        return list(s)
        
        def removeDuplicates(self, nums):
            #26. Remove Duplicates from Sorted Array
            if len(nums)==0:
                return 0
            idx = 0
            for i in xrange(1,len(nums)):
                if nums[idx]!=nums[i]:
                    idx = idx + 1
                    nums[idx]=nums[i]
            return idx + 1
        
    def myPow(self, x, n):
        #50. Pow(x, n)
        if n == 0:
            return 1
        if n ==-1:
            return 1/x
        return self.myPow(x*x, n/2) * [1,x][n % 2]

    def setZeroes(self, matrix):
        #73. Set Matrix Zeroes
        # CCI 1.8
        cols = set()
        rows = set()
        zeroes = set()
        for i,r in enumerate(matrix):
            for j,c in enumerate(r):
                if matrix[i][j]==0:
                    rows.add(i)
                    cols.add(j)
        for i in rows:
            for j,_ in enumerate(matrix[i]):
                zeroes.add((i,j))
        for j in cols:
            for i in xrange(len(matrix)):
                zeroes.add((i,j))
        for i,j in zeroes:
            matrix[i][j]=0

    def isScramble(self, s1, s2):
        #87. Scramble String
        if s1==s2:
            return True
        if len(s1)!=len(s2):
            return False
        from collections import Counter
        if Counter(s1)!=Counter(s2):
            return False
        elif len(s1)<4:
            return True
        for i in xrange(1,len(s1)):
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]):
                return True
            if self.isScramble(s1[:i], s2[-i:]) and self.isScramble(s1[i:], s2[:-i]):
                return True
        return False

    def longestPalindrome(self,s):
        n = len(s)
        #lookup Table
        L = [[0 for x in range(n)] for x in range(n)]
        #singleton palendroms
        for i in range(n):
            L[i][i]=1
        #itterate solution
        for lps in range(2,n+1):
            for i in range(n-lps+1):
                j = i + lps-1
                if s[i] == s[j] and lps == 2:
                    L[i][j] = 2
                elif s[i] == s[j]:
                    L[i][j] = L[i+1][j-1] + 2
                else:
                    L[i][j] = max(L[i][j-1], L[i+1][j])
        return L[0][n-1]

    def isUnique(self,s):
        ## determine if a strhing has all unique characters
        ## CC 1.1
        seen = set()
        for letter in s:
            if letter in seen:
                return False
            else:
                seen.add(letter)
        return True

    def isUniqueNoDataStructures(self,s):
        ## CC 1.1b
        s = sorted(s)
        for i,letter in enumerate(s):
            if i == len(s)-2:
                if letter == s[i+1]:
                    return False
                else:
                    return True
            elif letter == s[i+1]:
                return False
        return True
            
    def checkPermutation(self, s1, s2):
        ## given 2 strings, decide if one is a permutation of the other
        ## CC 1.2
        if len(s1) != len(s2):
            return False
        from collections import Counter
        if Counter(s1) == Counter(s2):
            return True
        return False

    def URLify(self, s):
        ## CC 1.3
        return ''.join( ['%20' if letter==' ' else letter for letter in s.rstrip()])

    def palindromePermutation(self, s):
        ### CC 1.4
        s = s.replace(' ', '').lower()
        from collections import Counter
        d = Counter(s)
        if len(s)%2 == 0:
            odds_allowed = 0
        else:
            odds_allowed = 1
        ct = 0
        for key, value in d.iteritems():
            if value %2 != 0:
                ct = ct + 1
            if ct > odds_allowed:
                return False
        return True

    def oneAway(self, s1, s2):
        ### CC 1.5
        if abs(len(s1)-len(s2))>=2:
            return False
        ## insert
        differences = 0
        offset = 0
        if len(s1)>len(s2):
            small = s2
            big = s1
        ## remove
        elif len(s1)<len(s2):
            small = s1
            big = s2
        ## replace
        else:
            d = {}
            for letter, letter2 in zip(s1,s2):
                if letter!=letter2:
                    if len(d)==0:
                        d[letter]=letter2
                    elif len(d)==1:
                        l = d.get(letter,'<unk>')
                        if l!=letter2:
                            return False
                    else:
                        return False
            return True
        for i, letter in enumerate(small):
            if big[i+offset]!=letter:
                differences+=1
                offset=1
            if differences>1:
                return False
        return True

    def stringCompression(self,s):
        ### CC 1.6
        r = []
        ct = 1
        for i, l in enumerate(s):
            if i+1 < len(s) and l==s[i+1]:
                ct = ct + 1
            else:
                r.append(l + str(ct))
                ct = 1
        if len(s)>len(r):
            return ''.join(r)
        else:
            return s

    def rotateMatrix(self, matrix):
        ## CC 1.7
        return [list(m) for m in zip(*matrix[::-1])]

    def stringRotation(self,s1,s2):
        ## CC 1.8
        if len(s1)!=len(s2) != 0:
            return False
        s1 = s1 + s1
        return s2 in s1

    def removeDupes(self, l1=[]):
        s=set()
        return [(this,s.add(this))[0]  for this in l1 if this not in s]

    def countBits(self, num):
        return ["{0:b}".format(i).count('1') for i in xrange(num+1)]

    def convert(self, s, numRows):
        #6. ZigZag Conversion https://leetcode.com/problems/zigzag-conversion/#/description
        lists = [ [] for i in xrange(n)]
        r, c = 0,0
        for v in s:
            if r == numRows-1:
                r=0
                c = c-1

    def reverseWords(self, s):
        #151. Reverse Words in a String https://leetcode.com/problems/reverse-words-in-a-string/#/description
        return ' '.join(s.strip().split()[::-1])

    def findMin(self, nums):
        #154. Find Minimum in Rotated Sorted Array II https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/#/description
        if len(nums)==1:
            return nums[0]
        elif len(nums)==2:
            return min(nums[0],nums[1])
        left = nums[0]
        right = nums[len(nums)-1]
        mid = nums[len(nums)/2]
        if left<right:
            return nums[0]
        elif mid > left or mid>right:
            return self.findMin(nums[len(nums)/2:])
        elif mid == left and mid == right:
            return min( self.findMin(nums[:len(nums)/2+1]), self.findMin(nums[len(nums)/2:]) )
        else:
            return self.findMin(nums[:len(nums)/2+1])

    def maximumGap(self, nums):
        #164. Maximum Gap https://leetcode.com/problems/maximum-gap/#/description
        if len(nums)<=1:
            return 0
        mx = -2147483647
        nums = sorted(nums)
        for i in xrange(1, len(nums) ):
            mx = max(mx, nums[i] - nums[i-1] )
        return mx

    def containsNearbyDuplicate(self, nums, k):
        #219. Contains Duplicate II https://leetcode.com/problems/contains-duplicate-ii/#/description
        #Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array
        #st nums[i] = nums[j] and the absolute difference between i and j is at most k.
        from collections import defaultdict
        d = {}
        d = defaultdict(lambda:-k - 1, d)
        for i,v in enumerate(nums):
            if d[v] >=i:
                return True
            d[v] = i + k
        return False

    def minDepth(self, root):
        # 111. Minimum Depth of Binary Tree
        # https://leetcode.com/problems/minimum-depth-of-binary-tree/#/description
        # Given a binary tree, find its minimum depth.
        #The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
        pass
    
    def maxArea(self, height):
        water = 0
        for i, v in enumerate(height):
            if i+1!=len(height):
                water = water + min(height[i],height[i+1])
        return water

    def RemoveDupes(self,ll):
        ## CCI 2.1 removed duplicates from an unsorted linked list
        if ll.head is None:
            return ll
        current = ll.head
        s = set([current.value])
        while current.next:
            if current.next.value in s:
                current.next = current.next.next
            else:
                s.add(current.value)
                current = current.next
        return ll

    def removeDupesNoBuffer(self,ll):
        ## CCI 2.1.b removed duplicates from an unsorted linked list, no buffer - use runner
        if ll.head is None:
            return ll
        current = ll.head
        while current.next:
            ## check for duplicate
            runner = current
            while runner.next:
                if runner.next.value == current.value:
                    runner.next = runner.next.next
                else:
                    runner = runner.next
            current = current.next
        return ll

    def printKthToLast(self,ll, k):
        count = 0
        current = ll.head
        while current.next:
            count += 1
            current = current.next
        v = count - k + 1
        current = ll.head
        count = 0
        while current.next:
            if count == v:
                return current.value
            current = current.next
            count += 1
        return current.value

    def getHint(self, secret, guess):
        #299. Bulls and Cows
        from collections import Counter
        s = Counter(secret)
        g = Counter(guess)
        bulls = sum( [i == j for i, j in zip(secret,guess) ] )
        cows = sum(  (s & g).values()) - bulls
        return str(bulls) + 'A' + str(cows) + 'B'

    def shortedNdistinct(self,s1,s2):
        # Given 2 arrays sorted and distinct, find common elements
        # *** still working on this
        i, j = 0, 0
        s = []
        while i < len(s1)-1 or j < len(s2)-1:
            if s1[i]==s2[j]:
                s.append(s1[i])
                i = min(i + 1, len(s1)-1)
                j = min(j + 1, len(s2)-1)
            if i< len(s1) and s1[i] < s2[j]:
                i = min(i + 1, len(s1)-1)
            if j< len(s2) and s1[i] > s2[j] :
                j = min(j + 1, len(s2)-1)
        return s

    def findMinMoves(self, machines):
        pass
            
        


s= Solution()

print s.findMinMoves([1,0,5])

#print (s.shortedNdistinct([1,2,3,5,10],[2,10]))

#ll = LinkedList([1,1,2,3,4,5,5])
#ll = LinkedList([5,1,3,4,5,2,1])




