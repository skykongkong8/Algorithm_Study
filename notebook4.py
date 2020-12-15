"""15650 백트래킹 알고리즘 -- N과 M (2)"""
# from itertools import combinations
# N, M = map(int, input().split())
# res_str = ""
# #join은 튜플을 문자열로 바꿀때도 사용된다. 만약 내부가 int라면, map을 사용하면 된다!
# for num in list(combinations([x for x in range(1, N+1)], M)):
#     res_str += " ".join(map(str, num))
#     res_str +='\n'
# print(res_str)
        
#다른 풀이 --실제 백트래킹 알고리즘 활용
# N, M = map(int, input().split()) #입력 부분
# visited = [False] * N #False로 작성된 불리언
out, out_all = [], [] #해당 조건을 하나씩 담아줄 리스트 out과 이들을 한꺼번에 출력하기 위해 res_str 같은 역할의 out_all

# def solve(depth, N, M):
#     if depth == M:
#         out_str = ' '.join(map(str, sorted(out)))
#         if out_str not in out_all:
#             out_all.append(out_str)
#         return
#     for i in range(N): #각각의 수 별로:
#         if not visited[i]:
#             visited[i] = True
#             out.append(i+1)
#             solve(depth+1, N, M)
#             visited[i] = False
#             out.pop()
# solve(0, N, M)

# for i in out_all:
#     print(i)


"""15651 N과 M (3)"""
# from itertools import product
# N, M = map(int, input().split())
# res_str = ""
# #join은 튜플을 문자열로 바꿀때도 사용된다. 만약 내부가 int라면, map을 사용하면 된다!
# for num in product([x for x in range(1, N+1)], M):
#     res_str += " ".join(map(str, num))
#     res_str +='\n'
# print(res_str)

"""---백트래킹 알고리즘 왤캐 어렵나!----"""

"""Dynamic Programming 알고리즘 101"""

"""1003 피보나치 함수의 베이스 케이스의 호출 횟수를 0.25초 안에 계산하는 알고리즘"""
"""int fibonacci(int n) {
    if (n == 0) {
        printf("0");
        return 0;
    } else if (n == 1) {
        printf("1");
        return 1;
    } else {
        return fibonacci(n‐1) + fibonacci(n‐2);
    }
}"""#사용하는 피보나치 함수는 C++ 문법으로 적혀있다.

# import sys
# N = int(sys.stdin.readline())
# res_str = ""

# num_list = []
# for _ in range(N):
#     num_list.append(int(sys.stdin.readline()))

def fibbonachi_zero_one_calls(num_list):
    dynamic_list = [[1,0],[0,1]]
    for i in range(2,max(num_list)+1):
        dynamic_list.append([dynamic_list[i-1][0]+dynamic_list[i-2][0], dynamic_list[i-1][1]+dynamic_list[i-2][1]])
    
    res_str = ""
    for n in num_list:
        res_str += f"{dynamic_list[n][0]} {dynamic_list[n][1]}"
        res_str += "\n"

    return res_str
# print(fibbonachi_zero_one_calls(num_list))

"""1904 01타일"""
# import sys
# N = int(sys.stdin.readline())

def zero_one_tile(N):
    tabulation_list = [0,1,2]
    for i in range(3, N+1):
        tabulation_list.append((tabulation_list[i-1]+tabulation_list[i-2])%15746)
        
    
    return tabulation_list[N]
# print(zero_one_tile(N))

# 팁: int형도 어떤것의 나머지로 저장하면 자리수가 적어지면서 메모리가 아껴진다!

"""RGB 거리 1149"""

import sys
import copy
class House:
    def __init__(self, price_list):
        self.price_list = price_list
        #클래스 인스턴스끼리 연결
        self.after = None

    def set_neighbors(self,house_after):
        self.after = house_after
        
#클래스 인스턴스 형성과 앞뒤 관계 설정
# House_list = [House([759,85,260]),House([136,226,532]),House([201,3,959]),House([132, 607, 359]),House([601, 775, 848]),House([462, 776, 920]),House([74,807,671])]
"""N = int(sys.stdin.readline())
House_list = []
for i in range(N):
    R,G,B = map(int, sys.stdin.readline().split())
    House_list.append(House([R,G,B]))

for i in range(len(House_list)-1):
    House_list[i].set_neighbors(House_list[i+1])

#가격 비교
#첫 번째
entire_price =0
color_before = None 
true_color_before = None 

second_break = False
for X in range(len(House_list)-1):
    for i in range(3):
        if i != true_color_before:
            for j in range(3):
                if j != i:
                    price = House_list[X].price_list[i]+House_list[j].after.price_list[1]
                    second_break = True
                    break
        if second_break:
            temp_color_before = i
            break
    first_price = copy.deepcopy(price)
    temp_price = price
    

    for i in range(3):
        if i != true_color_before:
            for j in range(3):
                if j != i:
                    temp_price = House_list[X].price_list[i]+House_list[X].after.price_list[j]
                else:
                    continue
                price_before = copy.deepcopy(price)
                price = min(price, temp_price)
                if price != price_before: #바뀌었다면
                    color_before = i   #사용한 색상의 순서를 저장
    
    
    if price == first_price:
        true_color_before = temp_color_before
    else:
        true_color_before = copy.deepcopy(color_before)
    print(House_list[X].price_list[true_color_before])
    entire_price += House_list[X].price_list[true_color_before]
    second_break = False

#마지막

for i in range(3):
    if i!=color_before:
        min_price = House_list[-1].price_list[i]
        break

for i in range(3):
    if i!= color_before:
        min_price = min(min_price, House_list[-1].price_list[i])
entire_price += min_price

print(min_price)
print(entire_price)"""

#우려했던 일이 현실로: Greedy 하게? 하면 나중에 아닐수도있음!예) 극단적인 차이가 날 경우
#완전 탐색 알고리즘을 짜보자.

"""1982 정수삼각형 --- DP Memoization 진짜 간G"""
# import sys
# sys.setrecursionlimit(10**6)
# def parse():
#     return list(map(int, sys.stdin.readline().split()))

# n, = parse()
# int_triangle = [[int(sys.stdin.readline())]] #처음꺼는 한자리니까 따로, 리스트 꼴이어야하니 리스트에 담아놓은 형태
# for _ in range(1, n):
#     int_triangle.append(parse())
# print(int_triangle)
#하위 항목들의 값을 재귀적으로 분석하면서 저장할 리스트
# memo = [[None]*i for i in range(1,n+1)]

# def best_route(floor, index):
#     #베이스 케이스-- 인셉션 탈출
#     if floor == n:
#         return 0
#     if memo[floor][index] is not None:
#         return memo[floor][index]
#     temp = []
#     for i in [index, index+1]:
#         temp.append(best_route(floor+1, i)+int_triangle[floor][index])
#     temp = max(temp)

#     memo[floor][index] = temp
#     return temp
# print(best_route(0,0))

"""2579 계단 오르기 DP"""
# import sys
# N = int(sys.stdin.readline())
# sys.setrecursionlimit(10**6)
# stairs = []

# for _ in range(N):
#     stairs.append(int(sys.stdin.readline()))
# N=6
# stairs = [6,10,20,15,25,10,20]

# #1경우의 수를 맞출 수 없다면 최대 경우의 수만큼 메모를 None으로 채워놓자?
# memo = [[None]*N for i in range(1,N+1)]
# def climb_stairs(floor, before):
#     if floor == 1:
#         return stairs[0]
#     if floor ==0:
#         return 0
#     if memo[floor][before] is not None:
#         return memo[floor][before]

#     continuity =0
#     for s in [floor-1, floor-2]:
#         for b in [s-1, s-2]:
#             if s>0 and b>0:
#                 if abs(b-s) ==1:
#                     continuity +=1
#                     if continuity >=3:
#                         break
                
#                 ret = (climb_stairs(s, b)+stairs[floor])
#                 memo[s][b] = ret
#     return max(memo[floor])


# print(max(climb_stairs(N-1,N-2), climb_stairs(N-1, N-3)))
# print(memo)


# """위 문제 해설 -- 재귀적 구조를 좀 더 조사하기"""
# import sys
# N = int(sys.stdin.readline())
# stairs = [int(sys.stdin.readline()) for _ in range(N)]
# stairs.insert(0,0)
# memo = [[0,0] for _ in range(N+1)]

# def stair_climbing(N):
#     for i in range(N+1):
#         if i == 1:
#             memo[1][0] = stairs[1]
#             memo[1][1] = stairs[1] #한칸전에서 왔던, 두 칸 전에서 왔던, 초기값임. '1번칸을 밟은 것'이라고 통일화된 개념이다. 
#         else:
#             memo[i][0] = memo[i-1][1] + stairs[i]
#             memo[i][1] = max(memo[i-2]) + stairs[i]
#     return max(memo[N])

# print(stair_climbing(N))


"""1463 1로 만들기 DP"""
# import sys
# sys.setrecursionlimit(10**6)
# N = int(sys.stdin.readline())

# memo = [None] * (N-1)
# memo.insert(0,0)

# def making_one(N):
    
#     if N == 1:
#         return 0
#     if memo[N-1] is not None:
#         return memo[N-1]
    
#     for i in range(2,N+1):
#         temp = []
#         if i%3 == 0:
#             temp.append(1 + making_one(i//3))
        
#         if (i-1)%3 ==0:
#             temp.append(2+making_one((i-1)//3))
#         if i%2 == 0:
#             temp.append(1+making_one(i//2))
#         if (i-1)%2==0:
#             temp.append(2+making_one((i-1)//2))
#         if (i-2)%3 == 0:
#             if i !=2:
#                 temp.append(3+making_one((i-2)//3))


#         temp = min(temp)
#         try:
#             memo[i-1] = min(temp, memo[i-1]) 
#         except:
#             memo[i-1] = temp 
#     return memo[N-1]
# print(making_one(N))

"""10844 계단수"""

#1 아래의 경우는 시뮬레이션 자체를 이용하여 풀이한 경우. 시뮬레이션을 이용하지 않고 DP로 풀어야 문제를 풀 수 있다. 10억까지 있는 큰 데이터의 문제이기 때문

# relation_list = [[1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9],[8]]#각 인덱스: 수, 저장된 리스트: 다음에 올수있는 수의 경우
# import sys
# N = int(sys.stdin.readline())

# class Tree:
#     def __init__(self, number):
#         self.number = number
#         self.child_list = []
#         self.depth = 0
#         self.relation = relation_list[number]

# Tree_list = [Tree(i) for i in range(1,10)]
# depth = 1

# if N != 1:
#     while depth<N+1:

#         for i in range(len(Tree_list)):
#             Tree_list[i].depth = depth
#             for j in range(len(Tree_list[i].relation)):
#                 Tree_list[i].child_list.append(Tree(Tree_list[i].relation[j]))
        
#         for tree in Tree_list:
#             for j in range(len(tree.child_list)):
#                 Tree_list.append(tree.child_list[j])
#             if tree.depth == depth:
#                 Tree_list.remove(tree)
#         depth+=1
    
#     count=0
#     for tree in Tree_list:
#         for i in range(len(tree.child_list)):
#             print(tree.child_list[i].number)
#             count+=1
#             count%=100000000
#     print(count%1000000000)
# else:
#     print(len(Tree_list))

""""'쉬운' 계단 수 DP 쉽긴뭐가시워,,,, 10844"""
#풀이는 세상 간단하다. DP의 기본원리는 늘 변하지 않는다. 베이스 케이스와 재귀규칙.
# import sys
# N = int(sys.stdin.readline())
# memo = [[],[1,1,1,1,1,1,1,1,1,1]] #빈 리스트 []는 길이의 표면적 값과 인덱스값을 일치시켜주기 위함이다. 0도 포함해주어야 1의 숫자를 셀때 제대로 세어지기 때문에 1로 처리해주어야 한다.
# standard = 1000000000

# for i in range(2, N+1): 
#     temp=[]
#     for j in range(10): 
#         if j ==0: 
#             temp.append(memo[i-1][1]%standard)
#         elif j ==9:
#             temp.append(memo[i-1][8]%standard)
#         else:
#             temp.append(memo[i-1][j-1]%standard+memo[i-1][j+1]%standard)
#     memo.append(temp)
# print((sum(memo[N])-memo[N][0])%standard)


"""2156 포도주 DP"""
# import sys
# N = int(sys.stdin.readline())
# wine = [int(sys.stdin.readline()) for _ in range(N)]
# # N =6
# # wine=[999,999,1,1,999,999]
# wine.insert(0,0)
# memo1 = [[0,0,0] for _ in range(N+1)]
# memo2 = [[0,0,0] for _ in range(N)]
# memo3 = [[0,0,0] for _ in range(N-1)]
# all_memo = [memo1, memo2, memo3]

# def greedy_drinker(K):
#     m = N-K
#     for i in range(K+1):#N을 넣으면 N까지 탐색, N-1을 넣으면 N-1까지 탐색---
#         if i ==1:
#             all_memo[m][i][0] = wine[1]
#             all_memo[m][i][1] = wine[1]
#             if K>=3:
#                 all_memo[m][i][2] = wine[1]
        
#         if i ==2:
#             all_memo[m][i][0] = wine[2]+wine[1]
#             all_memo[m][i][1] = wine[2]
#             if K>=3:
#                 all_memo[m][i][2] = wine[2]

#         else:  
#             all_memo[m][i][0] = max(all_memo[m][i-1][1] + wine[i],all_memo[m][i-1][2] + wine[i])
#             all_memo[m][i][1] = max(all_memo[m][i-2]) + wine[i]
#             if K>=3:
#                 all_memo[m][i][2] = max(all_memo[m][i-3]) + wine[i]
#
#     return max(all_memo[m][K])
# if N>2:
#     print(max(greedy_drinker(N), greedy_drinker(N-1), greedy_drinker(N-2)))
# else:
#     print(sum(wine)) #병이 2개뿐이라면, 그냥 전체다 마시면 된다. ++ 병이 3개만이라면, 3칸띄우기 조건을 고려하지 않아도 된다.

"""11053 가징 긴 증가하는 부분 수열 DP"""
# import sys
# # N = int(sys.stdin.readline())
# series = list(map(int, sys.stdin.readline().split()))

# key_list = []
# for key in set(series):
#     key_list.append(series.index(key))
# print(key_list)
# key_list.sort()
# real_key=[]
# for i in range(1,len(key_list)):
#     Boo = True
#     for j in range(i):
#         if series[key_list[i]] >= series[key_list[j]]:
#             Boo = False
#             break
#     if Boo:
#         real_key.append(key_list[i])

# real_key.append(key_list[0])
# print(real_key)
# #위의 코드로 입력받은 수열에서 중복되지 않는 애들중, 최소의 위치의 있는 애들만 데려왔다.

# memo = 0
# for index in real_key:

#     # iterator = series[index]
#     # count = 1
#     for num in series[index:]:
#         count = 1
#         iterator = series[index]
#         for sub_num in series[index:][series[index:].index(num):]:
#             if sub_num>iterator:
#                 count+=1
#                 iterator = sub_num
#     memo = max(memo, count)
# print(memo)
 

"""위 문제의 기본 개념 알고리즘--LIS알고리즘"""
#Longest Increasing Subsequence
#1 완전탐색 시간복잡도 O(2^N)
import sys
# arr = list(map(int, sys.stdin.readline()))

def LIS_1(arr):
    if not arr:
        return 0

    ret = 1
    for i in range(len(arr)):
        nxt = []
        for j in range(i+1, len(arr)): #0번 인덱스 기준으로 조사 -- 그다음 0번 인덱스를 제외한 1번인덱스 조사-- 2번--3번 중 제일 큰거..
            if arr[i] < arr[j]:
                nxt.append(arr[j])
        ret = max(ret, 1+LIS_1(nxt))
        
    return ret

#2 동적 계획법 적용하기(원리는 같으나 캐싱을 사용한다) O(N**2)

import math
def LIS_2(arr):
    arr = [-math.inf] + arr
    N = len(arr)
    cache = [-1] * N

    def find(start):
        if cache[start] != -1:
            return cache[start]

        ret = 0
        for nxt in range(start+1, N):
            if arr[start] < arr[nxt]:
                ret = max(ret, find(nxt) + 1) #알고리즘은 똑같은데

        cache[start] = ret#이렇게 캐싱을 주었다.
        return ret

    return find(0)

#3 이진 탐색을 활용하기 O(NlgN)
def LIS_3(arr):
    if not arr:
        return 0

    # C[i] means smallest last number of lis subsequences whose length are i
    INF = float('inf')
    C = [INF] * (len(arr)+1)
    C[0] = -INF
    C[1] = arr[0]
    tmp_longest = 1

    # Find i that matches C[i-1] < n <= C[i]
    def search(lo, hi, n):
        if lo == hi:
            return lo
        elif lo + 1 == hi:
            return lo if C[lo] >= n else hi

        mid = (lo + hi) // 2
        if C[mid] == n:
            return mid
        elif C[mid] < n:
            return search(mid+1, hi, n)
        else:
            return search(lo, mid, n)


    for n in arr:
        if C[tmp_longest] < n:
            tmp_longest += 1
            C[tmp_longest] = n
        else:
            next_loc = search(0, tmp_longest, n)
            C[next_loc] = n

    return tmp_longest

"""LIS 알고리즘을 잘 활용하기"""
"""가장 긴 바이토닉 부분 수열 11054"""

"""DP,, 이젠 못하겠어,,"""


"""그리디 알고리즘"""

"""11399"""
import heapq
# import sys
# N = int(sys.stdin.readline())
# arr = list(map(int, sys.stdin.readline().split()))


def ATM(arr):
    time =0
    count =1
    arr.sort()
    while arr:
        time += count*arr.pop()
        count+=1
    return time
# print(ATM(arr))

"""1541 잃어버린 괄호"""
# import sys
# my_str = sys.stdin.readline()
# plus_sum=0
# minus_sum=0

# if '-' not in my_str:
#     word_split = my_str.split('+')
#     for k in word_split:
#         plus_sum+=int(k)
# if '-' in my_str:
#     minus_plus_split = my_str.split('-')
#     # print(minus_plus_split)
#     for i in range(len(minus_plus_split)):
#         if i ==0:
#             try:
#                 plus_sum += int(minus_plus_split[0])
#             except:
#                 one_can_be_like_this = minus_plus_split[0].split('+')
#                 for i in range(len(one_can_be_like_this)):
#                     plus_sum+=int(one_can_be_like_this[i])
                    

#         else:
#             try:
#                 minus_sum += int(minus_plus_split[i])
#             except:
#                 minus_list = minus_plus_split[i].split('+')
#                 for m in minus_list:
#                     minus_sum += int(m)
# # print(plus_sum)    
# # print(minus_sum)
# print(plus_sum-minus_sum)
    


"""1931 회의실 배정 Greedy Algorithmn"""
# import sys
# N = int(sys.stdin.readline())
# meeting = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
# # meeting = [(1, 4),(3, 5),(0, 6),(5, 7),(3, 8),(5, 9),(6, 10),(8, 11),(8, 12),(2, 13),(12, 14)]
# print(meeting)
# meeting.sort(key = lambda x: (x[1]-x[0], x[0], x[1]))
# print(meeting)

# # meeting_room = [0]*(2**31-1)
# meeting_room = {}

# count =0
# for meet in meeting:
#     Boo = True
#     for i in range(meet[0], meet[1]):
#         if i in meeting_room:
#             Boo = False
#             break
#         elif i not in meeting_room:
#             meeting_room[i] = 1
        
#     if Boo:
#         count+=1
# print(count)

"""스택 구현"""

"""10828 스택"""
from collections import deque
import sys
import copy
# stack = deque()

# N = int(sys.stdin.readline())
# for i in range(N):
#     order = sys.stdin.readline()

#     if 'push' in order:
#         order_push = order.split(' ')
#         stack.append(int(order_push[1]))
    
#     elif order == 'top\n':
#         if stack:
#            dup = copy.deepcopy(stack)
#            print(dup.pop())
#         else:
#             print(-1)
    
#     elif order == 'size\n':
#         count=0
#         dup = copy.deepcopy(stack)
#         while dup:
#             dup.pop()
#             count+=1
#         print(count)
    
#     elif order == 'empty\n':
#         if stack:
#             print(0)
#         else:
#             print(1)
    
#     elif order == 'pop\n':
#         if stack:
#             print(stack.pop())
#         else:
#             print(-1)


"""10773 제로 Stack"""
# from collections import deque
# stack = deque()
# import sys
# N = int(sys.stdin.readline())
# for i in range(N):
#     num = int(sys.stdin.readline())
#     if num != 0:
#         stack.append(num)
#     else:
#         stack.pop()
# hab =0
# while stack:
#     hab+=stack.pop()
# print(hab)


"""9012 올바른 괄호, Stack"""
# from collections import deque
# import sys
# N = int(sys.stdin.readline())
# res_str = ""

# for _ in range(N):
#     parenthesis = sys.stdin.readline()
#     stack = deque()
#     Boo = False
#     for p in parenthesis:
#         if p == '(':
#             Boo = True
#             stack.append(p)
#         elif p == ')':
#             try:
#                 stack.pop()
#             except:
#                 res_str += 'NO\n'
#                 Boo = False
#                 break
#     if Boo:
#         if not stack:
#             res_str += "YES\n"
#         elif stack:
#             res_str += 'NO\n'
#     else:
#         if stack:
#             res_str += 'NO\n'

# print(res_str)


"""4949 균형 잡힌 세상 -- even more complicated parenthesis Stack"""
# import sys
# from collections import deque
# # string = sys.stdin.readline().split('.')
# string = 'So when I die (the [first] I will see in (heaven) is a score list).[ first in ] ( first out ).Half Moon tonight (At least it is better than no Moon at all].A rope may form )( a trail in a maze.Help( I[m being held prisoner in a fortune cookie factory)].([ (([( [ ] ) ( ) (( ))] )) ]). ..'.split('.')


# for rat in string:
#     stack1 = deque(); Boo1 = False; BooC=0
#     stack2 = deque(); Boo2 = False; BooE=0
    
#     #1 그냥 . 만 있는 경우
#     if rat == ' ':
#         print('yes')
#         continue
    
#     for p in rat:
#         #2 () 소괄호 검사
#         if p == '(':
#             stack1.append(p)
#             Boo1 = True
#             BooC +=1
#         elif p==')':
#             try:
#                 stack1.pop()
#                 BooC-=1
#             except:
#                 # print('no')
#                 Boo1 = False
#                 break
        
#         #3 [] 대괄호 검사
#         elif p == '[':
#             stack2.append(p)
#             Boo2 = True
#             BooE+=1
#         elif p ==']':
#             try:
#                 stack2.pop()
#                 BooE-=1
                

#             except:
#                 # print('no')
#                 Boo2 = False
#                 break
        
#     #4 최종 검사
#     if Boo1 and Boo2:
#         if not stack1:
#             if not stack2:
#                 print('yes')
#     else:
#         if rat != '':
#             print('no')



# import sys
# from collections import deque
# # string = sys.stdin.readline().split('.')
# string = 'So when I die (the [first] I will see in (heaven) is a score list).[ first in ] ( first out ).Half Moon tonight (At least it is better than no Moon at all].A rope may form )( a trail in a maze.Help( I[m being held prisoner in a fortune cookie factory)].([ (([( [ ] ) ( ) (( ))] )) ]). ..'.split('.')
# # string=['Help( I[m being held prisoner in a fortune cookie factory)]']


def The_test(rat):
    if C_test(rat) and E_test(rat):
        print('yes')
    else:
        if rat != '':
            print('no')


def C_test(rat):
    Boo1 = True
    stack1 = deque()    
    for i in range(len(rat)):
        #2 () 소괄호 검사
        if rat[i] == '(':
            stack1.append(rat[i])
            Boo1 = True
        elif rat[i]==')':
            try:
                stack1.pop()
            except:
                # print('no')
                Boo1 = False
                break
        elif rat[i]=='[' or rat[i]==']':
            if E_test(rat[i:]):
               pass
            else:
                Boo1=False
                break

    if Boo1:
        if not stack1:
            return True
    else:
        return False

def E_test(rat):
    Boo2 = True
    stack2 = deque()
    for j in range(len(rat)):        
        #3 [] 대괄호 검사
        if rat[j] == '[':
            stack2.append(rat[j])
            Boo2 = True
        elif rat[j] ==']':
            try:
                stack2.pop()
            except:
                # print('no')
                Boo2 = False
                break
        elif rat[j]=='(' or rat[j]==')':
            if C_test(rat[j:]):
               pass
            else:
                Boo2=False
                break
    if Boo2:
        if not stack2:
            return True
    else:
        return False


# for rat in string:    
#     #1 그냥 . 만 있는 경우
#     if rat == ' ':
#         print('yes')
#         continue
#     else:
#         The_test(rat)


"""18258 큐2"""
# from collections import deque
# import sys
# queue = deque()

# N = int(sys.stdin.readline())
# for i in range(N):
#     order = sys.stdin.readline()

#     if 'push' in order:
#         order_push = order.split(' ')
#         queue.append(int(order_push[1]))
    
#     elif order == 'back\n':
#         if queue:
#            print(queue[-1])
#         else:
#             print(-1)

#     elif order == 'front\n':
#         if queue:
#            print(queue[0])
#         else:
#             print(-1)
    
#     elif order == 'size\n':
#         print(len(queue))
    
#     elif order == 'empty\n':
#         if queue:
#             print(0)
#         else:
#             print(1)
    
#     elif order == 'pop\n':
#         if queue:
#             print(queue.popleft())
#         else:
#             print(-1)


"""카드 2"""
# import sys
# N = int(sys.stdin.readline())
# from collections import deque
# queue = deque([i for i in range(1, N+1)])

def card_two(queue):
    while queue:
        if len(queue)>1:
            queue.popleft()
            queue.rotate(-1)
        if len(queue)<=1:
            break
    return queue.popleft()
# print(card_two(queue))


"""11866 요세푸스 문제 0"""
# import sys
# N, K = map(int, sys.stdin.readline().split())
# from collections import deque
# queue = deque([i for i in range(1, N+1)])
# res_str = "<"
# while queue:
#     queue.rotate(-(K-1))
#     res_str += str(queue.popleft())
#     res_str += ', '

# res_str = res_str[:-2]
# res_str+=">"
# print(res_str)


"""프린터 큐 1966"""
# import sys
# from collections import deque
# X = int(sys.stdin.readline())
# answer = []

# for _ in range(X):
#     N, M = map(int, sys.stdin.readline().split())    
#     importance_list = list(map(int, sys.stdin.readline().split()))
#     iterator = max(importance_list)
#     new_importance_list = list(enumerate(importance_list))
#     target = new_importance_list[M][0]
    
#     queue = deque(new_importance_list)
#     count =0
#     while queue:
#         if queue[0][1] >= iterator:
#             if queue[0][0] != target:
#                 queue.popleft()
#                 importance_list.remove(iterator)
#                 iterator = max(importance_list)
#                 count+=1
#             else:
#                 count+=1
#                 break
#         else:
#             queue.rotate(-1)
#     answer.append(count)
# for i in answer:
#     print(i)


"""덱 10866"""
# from collections import deque
# import sys
# deq = deque()

# N = int(sys.stdin.readline())
# for i in range(N):
#     order = sys.stdin.readline()

#     if 'push_front' in order:
#         order_push = order.split(' ')
#         deq.appendleft(int(order_push[1]))
    
#     elif 'push_back' in order:
#         order_push = order.split(' ')
#         deq.append(int(order_push[1]))
    
#     elif order == 'back\n':
#         if deq:
#            print(deq[-1])
#         else:
#             print(-1)

#     elif order == 'front\n':
#         if deq:
#            print(deq[0])
#         else:
#             print(-1)
    
#     elif order == 'size\n':
#         print(len(deq))
    
#     elif order == 'empty\n':
#         if deq:
#             print(0)
#         else:
#             print(1)
    
#     elif order == 'pop_front\n':
#         if deq:
#             print(deq.popleft())
#         else:
#             print(-1)
#     elif order == 'pop_back\n':
#         if deq:
#             print(deq.pop())
#         else:
#             print(-1)


"""1021 회전하는 큐"""
#주어진 순서대로 뽑아내기 -- 더쉽네
# import sys
# from collections import deque
# import copy

# A, B = map(int, sys.stdin.readline().split())
# target_list = list(map(int, sys.stdin.readline().split()))
# answer = 0
# current_num = 1
# queue = deque([i for i in range(1, A+1)])

# for target in target_list:
#     queue1 = copy.deepcopy(queue); count1=0
#     queue2= copy.deepcopy(queue); count2=0
#     if queue[0] == target:
#         queue.popleft()
#         continue
#     while queue1[0] != target:
#         queue1.rotate(1)
#         count1 +=1
#     while queue2[0] != target:
#         queue2.rotate(-1)
#         count2 +=1
#     if count1>count2:
#         answer += count2
#         queue = copy.deepcopy(queue2)
#     elif count1 <= count2:
#         answer += count1
#         queue = copy.deepcopy(queue1)
#     if queue[0] == target:
#         queue.popleft()
#         continue

# print(answer)


"""AC 5430"""
# import sys
# from collections import deque
# T = int(sys.stdin.readline())

# def AC(string, num_list,N):
#     num_list = deque(num_list)
#     count =0
#     for letter in string:
#         if letter =='R':
#             count+=1
#             # num_list.reverse()
#         elif letter =='D':
#             if count>0:
#                 if count%2==0:
#                     pass
#                 else:
#                     num_list.reverse()
#             count =0
#             num_list.popleft()
#     if count != 0:
#         if count%2==0:
#             pass
#         else:
#             num_list.reverse()  
#     return list(num_list)

# ANSWER = []
# for _ in range(T):
#     string = sys.stdin.readline()
#     n = int(sys.stdin.readline())
#     try:
#         num_list = list(map(int, input().strip('[').strip(']').split(',')))
#     except:
#         ANSWER.append('error')
#         continue
#     try:
#         ANSWER.append(AC(string, num_list,n))
#     except:
#         ANSWER.append('error')
# for i in ANSWER:
#     print(i)