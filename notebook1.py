# #백준 11047(그리디 알고리즘)
# """N종류의 동전을 각각 무한으로 가지고 있을때 가치의 주어진 K값을 구성할 수 있는 경우의 수를 출력하는 함수"""
# def coin_greedy(N, K, coin_list):
#     count = 0 #경우의 수를 저장하는 변수 count
#     for i in coin_list:
#         # if K-i<0:
#         #     pass
#         if K-i>=0:
#             count+=K//i
#             K = K%i #1은 무조건 있고, 동전 갯수에 제한이 없으므로!
#         if K == 0:
#             return count


# """각각의 동전 가짓수와 구성할 가치 입력받기"""    
# N,K = map(int, input().split())
# coin_list=[] #동전 가격이 저장된 리스트
# for i in range(N):
#     coin_list.append(int(input()))
# coin_list.reverse() #내림차순으로 정리하기
# print(coin_greedy(N,K, coin_list))
# # list1=[1,5,10,50,100,500,1000,5000,10000,50000]
# # list1.reverse()
# # print(coin_greedy(10,4790,list1))

#백준 1673

class Chicken_orderer:
    def __init__(self, coupon_num, needed_stamp):
        """가지고 있는 쿠폰의 개수, 필요한 도장 수, 현재 가지고 있는 도장, 현재 가지고 있는 치킨의 수"""
        self.coupon_num = coupon_num
        self.needed_stamp = needed_stamp
        self.stamp_now = 0
        self.chicken_now = 0
    
    def __str__(self):
        return str(self.chicken_now)

    def order_and_get_chicken(self):
        """쿠폰을 소모하여 치킨을 얻는 메소드"""
        self.chicken_now += self.coupon_num
        self.stamp_now += self.coupon_num

        self.coupon_num = 0
    
    def turn_stamp_into_coupon(self):
        if self.stamp_now > self.needed_stamp:
            self.coupon_num += self.stamp_now // self.needed_stamp
            self.stamp_now %= self.needed_stamp
        
        else:
            pass

# N,K = map(int, input().split())

# customer1 = Chicken_orderer(N, K)
# customer1.order_and_get_chicken()
# customer1.turn_stamp_into_coupon()
# while customer1.needed_stamp < customer1.stamp_now//customer1.needed_stamp or customer1.coupon_num > 0:
#     customer1.order_and_get_chicken()
#     customer1.turn_stamp_into_coupon()
#     customer1.order_and_get_chicken()

# print(customer1.chicken_now)
# ____chicken 문제의 문제 입력 부분_________


#큰 수 만들기(Greedy)

def solution(number, k):
    answer_list = []
    len_of_answer=len(number)-k
    indexes_left = len_of_answer
    while indexes_left != 0:
        #greedy적으로 n 자리 수라고 할 때, 첫째 자리 수가 가장 중요하므로 그 수를 찾는다
        #이 때, n 자리수를 뒤에 숫자 자릿수만큼 남겨둔 범위 내에서만 찾음
        if indexes_left == 1:
            target_number = number
        else:
            target_number=number[:-indexes_left+1]
        temp_list = list(map(int, target_number))
        if len(temp_list) != 0:
            my_number = max(temp_list)
            #그 다음 수는 첫째 자리 수 이후부터 위와 같은 맥락으로 다음 수를 찾는다
            answer_list.append(str(my_number))
            #number를 update, 남은 자릿수도 업데이트
            number = number[number.index(str(my_number))+1:]
            indexes_left -=1
    answer = ''.join(answer_list)        
    return answer

# print(solution('3234',2))



#프로그래머스 Lv2: 기능개발
"""queue 연습해보기"""
from collections import deque
# queue = deque()

# my_list = [1,2,3]
# my_list2 = [1,1,1]
# for i in my_list:
#     queue.append(i)
# print(queue)
# for i in range(len(queue)):
#     queue[i]+=my_list2[i]
# print(queue)

from collections import deque
def function_development(progresses, speeds):
    answer = []
    upgrade_queue = deque(progresses)
    speeds_queue = deque(speeds)
    while len(upgrade_queue) != 0:
        count = 0; index = 0
        for i in range(len(upgrade_queue)):
            upgrade_queue[i] += speeds_queue[i]
        try:
            while upgrade_queue[index] >= 100:
                index +=1
                count +=1
        except:
            pass
        for i in range(index):
            upgrade_queue.popleft()
            speeds_queue.popleft()
        if count > 0:
            answer.append(count)      
    return answer
# print(solution([93,30,55],[1,30,5]))

# 프로그래머스 Lv2 주식가격
from collections import deque
def Stockprices(prices):
    stock_queue = deque(prices)
    time_list=deque()
    while len(stock_queue) != 0:
        time = 0
        # if len(stock_queue) ==1:
        #     time_list.append(0)
        #     stock_queue.popleft() 
        # else:
        try:
            for i in range(1, len(stock_queue)):
                if stock_queue[0] <= stock_queue[i]:
                    time+=1
                else:
                    time+=1
                    break
        except:
            pass
            # time = min(time, len(stock_queue)-1)
            time_list.append(time)
            stock_queue.popleft()
    return list(time_list)

#프로그래머스 Lv2 다리를 지나는 트럭
from collections import deque
def crossing_trucks(bridge_length, weight, truck_weights):
    truck_queue = deque(truck_weights)
    bridge = deque()
    finished = deque()
    time =0
    while len(bridge) != 0 or len(truck_queue) != 0:
        while sum(bridge) <= weight and len(truck_queue)>0:
            bridge.append(truck_queue.popleft())
            time +=1
        
        if sum(bridge) > weight:
            truck_queue.appendleft(bridge.pop())
            time-=1
        while len(bridge) > 0:
            finished.append(bridge.popleft())
            time+=(bridge_length-1)
    return time

from collections import deque
def go(bridge, i):
    #단, i < bridge_length -1
    if bridge[i] != 0:
        temp = bridge[i]
        bridge[i]=bridge[i+1]
        bridge[i+1] = temp

crossing = [[1,1]]
bridge_length = 1



#프로그래머스 Lv2 스택/큐 프린터
# from collections import deque
# """enumerate 함수에 대한 연습"""
# list1 = [4,5,6,7,8]
# my_first_enumerate= deque(enumerate(list1))
# """마치 dictionary처럼 인덱스와 함께 튜플로 같이 묶인다. 이 정보를 한 층 더 활용할 수 있을듯"""
# #combinations와 같이 보통 list와 함께 쓰이는 경우가 일반적이지만, deque와도 쓸 수 있는듯!
# print(my_first_enumerate)
# #이러한 자료 구조형을 반복문과 같이 쓰기도 한다
# my_first_enumerate[1] = my_first_enumerate[2] #이렇게 인덱싱을 통해 자리를 바꿀 수 있다!
# print(my_first_enumerate)

def backswap(my_deque):
    """swap이랑 똑같지만 index1은 첫 번째거, index2는 마지막거를 써줄거임"""
    my_deque.append(my_deque.popleft())
from collections import deque
def printer(priorities, location):
    priorities_enumerate_queue = deque(enumerate(priorities))
    count = 0
    while len(priorities_enumerate_queue) != 0:
        max_priority_number = priorities_enumerate_queue[0][1]
        for i in range(len(priorities_enumerate_queue)-1):
            max_priority_number = max(max_priority_number, priorities_enumerate_queue[i][1])
        
        if priorities_enumerate_queue[0][1]!=max_priority_number:
            backswap(priorities_enumerate_queue)
        elif priorities_enumerate_queue[0][0] == location:
            count +=1
            break
        elif priorities_enumerate_queue[0][1]==max_priority_number:
            priorities_enumerate_queue.popleft()
            count+=1
        
       
    return count
# print(printer([2,1,3,2],2))
# print(printer([1,1,9,1,1,1],0))



#백준 아침워밍업 15596, 2675

def sum_of_N(N_list):
    return sum(N_list)
# print(sum_of_N([1,2,3,4,5]))

def sum_of_given(N, numbers):
    numbers = str(numbers).strip()
    answer = 0
    for i in range(len(numbers)):
        answer += int(numbers[i])
    return answer 
# N = int(input())
# num = int(input())
# print(sum_of_given(N,num))



#백준 알고리즘-브루트 포스
# 2750 수 정렬하기1,2,3 시간복잡도 n2 nlgn 카운팅 정렬
"""정수가 주어졌을 때, 이를 정렬해서 순서대로 출력하는 함수"""
# #N**2
# N = int(input())
# num_set = list()
# for i in range(N):
#     num_set.append(int(input()))
# num_set.sort()
# for i in range(N):
#     print(num_set[i])


#NlgN
# from collections import deque
# N = int(input())
# sorting_queue = deque()
# for i in range(N):
#     deque.append(int(input()))
#이진 탐색 트리를 in-order순회로 출력하여 크기 순으로 나열하기--런타임에러--보통 O(lgN)이라서 도합 NlgN일텐데..?
import time
import sys #sys.stdin.readline()을 사용하기 위해ㅎ--input보다 조금 더 빠르다
start = time.time()
class Node:
    def __init__(self, data):
        self.data = data
        self.parent = None
        self.left_child = None
        self.right_child = None
def in_order_traverse(node):
    if node is not None:
        in_order_traverse(node.left_child)
        print(node.data)
        in_order_traverse(node.right_child)
        
class BinarySearchTree:
    def __init__(self):
        self.root = None
    def insert(self, data):
        new_node = Node(data)
        if self.root is None:
            self.root = new_node
            return
        else:
            iterator = self.root
            iterator_just_before = None
            while iterator is not None:
                # iterator_just_before = iterator
                if iterator.data > new_node.data:
                    iterator_just_before = iterator
                    iterator = iterator.left_child
                else:
                    iterator_just_before = iterator 
                    iterator = iterator.right_child
            if iterator_just_before.data > new_node.data:
                iterator_just_before.left_child = new_node
            elif iterator_just_before.data < new_node.data:
                iterator_just_before.right_child = new_node
# N = int(sys.stdin.readline())
# bst = BinarySearchTree()
# for i in range(N):
#     bst.insert(int(sys.stdin.readline()))
# in_order_traverse(bst.root)
# print(f"time : {time.time()-start}")

#아이디어2-heapsort..? 근데 이건 완전이진트리속성을 먼저,,음,,
#시간 재기 코드
import time
start = time.time()

# """우선 완전 이진트리 형식을 갖추기 위해 리스트에 저장한다"""
# heap_list = [None]
# for i in range(int(sys.stdin.readline())):
#     heap_list.append(int(sys.stdin.readline()))
# tree_size = len(heap_list)-1(?)

"""heapsort를 수월하게 하기 위한 swap함수"""
def swap(tree, index1, index2):
    temp = tree[index1]
    tree[index1] = tree[index2]
    tree[index2] = temp

"""heapify 시키기를 위한 함수"""
def heapify(tree, index, tree_size):
    left_child_index = index*2
    right_child_index = index*2 + 1
    largest = index
    if (0<left_child_index<tree_size) and (tree[largest]<tree[left_child_index]):
        largest = left_child_index
    if (0<right_child_index<tree_size) and (tree[largest]<tree[right_child_index]):
        largest = right_child_index
    if largest != index:
        swap(tree, index, largest)
        heapify(tree, largest, tree_size)


"""본래의 heapsort 코드"""
def heapsort(tree):
    tree_size = len(tree)
    for index in range(tree_size, 0, -1):
        heapify(tree, index, tree_size)
    for i in range(tree_size-1, 0, -1): 
        swap(tree, 1, i)
        heapify(tree, 1, i)

"""정답을 계산하는 부분"""
# heap_list = [None]
# for i in range(int(input())):
#     heap_list.append(int(input())) ---이 부분을 위에서 진행함
# heapsort(heap_list)
# for i in range(1,len(heap_list)):
#     print(heap_list[i])
# # print(heap_list)
# print(f"time : {time.time()-start}")


class StationNode:
    """간단한 지하철 역 노드 클래스"""
    def __init__(self, station_name):
        self.station_name = station_name
        self.adjacent_stations = []  # 인접 리스트


    def add_connection(self, other_station):
        """지하철 역 노드 사이 엣지 저장하기"""
        self.adjacent_stations.append(other_station)
        other_station.adjacent_stations.append(self)


    def __str__(self):
        """지하철 노드 문자열 메소드. 지하철 역 이름과 연결된 역들을 모두 출력해준다"""
        res_str = f"{self.station_name}: "  # 리턴할 문자열

        # 리턴할 문자열에 인접한 역 이름들 저장
        for station in self.adjacent_stations:
            res_str += f"{station.station_name} "

        return res_str

# def create_subway_graph(input_file):
#     """input_file에서 데이터를 읽어 와서 지하철 그래프를 리턴하는 함수"""
#     stations = {}  # 지하철 역 노드들을 담을 딕셔너리

#     # 파라미터로 받은 input_file 파일을 연다
#     with open(input_file) as stations_raw_file:
#         for line in stations_raw_file:  # 파일을 한 줄씩 받아온다
#             previous_station = None  # 엣지를 저장하기 위한 도우미 변수. 현재 보고 있는 역 전 역을 저장한다
#             subway_line = line.strip().split("-")  # 앞 뒤 띄어쓰기를 없애고 "-"를 기준점으로 데이터를 나눈다

#             for name in subway_line:
#                 station_name = name.strip()  # 앞 뒤 띄어쓰기 없애기

#                 # 지하철 역 이름이 이미 저장한 key 인지 확인
#                 if station_name not in stations:
#                     current_station = StationNode(station_name)  # 새로운 인스턴스를 생성하고
#                     stations[station_name] = current_station  # dictionary에 역 이름은 key로, 역 인스턴스를 value로 저장한다

#                 else:
#                     current_station = stations[station_name]  # 이미 저장한 역이면 stations에서 역 인스턴스를 갖고 온다

#                 if previous_station is not None:
#                     current_station.add_connection(previous_station)  # 현재 역과 전 역의 엣지를 연결한다

#                 previous_station = current_station  # 현재 역을 전 역으로 저장

#     return stations


# stations = create_subway_graph("C:/Users/공성식/Desktop/WORKSTATION/Python Workplace/codeit.station_line.txt")  # stations.txt 파일로 그래프를 만든다

# stations에 저장한 역 인접 역들 출력 (체점을 위해 역 이름 순서대로 출력)
# for station in sorted(stations.keys()):
#         print(stations[station], stations) 

#백준 easies
# try:
#     N, my_string = input().split()
#     N=int(N)
#     real_string =''
#     for i in range(len(my_string)):
#         real_string += my_string[i]*N
#     print(real_string)
# except:
#     pass

#Dynamic Programming -- Fibbonachi
# answer_list =[]
# for _ in range(int(input())):
#     N = int(input())
#     dynamic_list = [1,1,1,2]
#     while len(dynamic_list) < N:
#         dynamic_list.append(dynamic_list[len(dynamic_list)-3]+ dynamic_list[len(dynamic_list)-2])
#     answer_list.append(dynamic_list[-1])
# # print(answer_list[i] for i in range(len(answer_list)))
# for i in answer_list:
#     print(i)

#전화번호 목록--해시
#def solution(phone_book):
#     for n in range(len(phone_book)):
#         phone_book[n] = int(phone_book[n])
#     phone_book.sort()
#     for n in range(len(phone_book)):
#         phone_book[n] = str(phone_book[n])
#     for i in range(len(phone_book)):
#         for j in phone_book[i+1:]:
#             if phone_book[i]==j[:len(phone_book[i])]:
#                 return False
#     else:
#         return True
# K=['1','123','56']
# for k in range(len(K)):
#     K[k] = int(K[k])
# K.sort()
# for k in range(len(K)):
#     K[k] = str(K[k])


#N개의 최소공배수
from math import gcd
def LCM(A,B):
    GCD = gcd(A,B)
    LCM = int(GCD*(A/GCD)*(B/GCD))
    return LCM
# print(LCM(2,3))

from collections import deque
def LCM_of_arr(arr):
    arr=deque(arr)
    while len(arr)>=2:
        A=arr[0]
        B=arr[1]
        GCD = gcd(A,B)
        LCM = int(GCD*(A/GCD)*(B/GCD))
        arr.popleft()
        arr.popleft()
        arr.append(LCM)
    return LCM
# print(LCM_of_arr([2,6,8,14]))


#위장
def number_of_basic_combinations(n):
    return (2**n) -1

from HDLL import LinkedList
from hash_table import HashTable
# wearing = LinkedList()
def camouflage(clothes):
    wearing = HashTable(len(clothes))
    for i in clothes:
        wearing.insert(i[1],i[0])
    camo = 1
    viewed_keys=[]
    for j in range(len(clothes)):
        if clothes[j][1] not in viewed_keys:
            camo *= (wearing.get_linked_list_for_key(clothes[j][1]).len()+1)
            viewed_keys.append(clothes[j][1])
    return camo-1
# print(camouflage([['yellow_hat', 'headgear'], ['blue_sunglasses', 'eyewear'], ['green_turban', 'headgear']]))


#백준 7568 '덩치'
class Fellas:
    def __init__(self, weight, height):
        self.weight = weight
        self.height = height
        self.rank = 1
    def is_bigger_than(self, other_fella):
        if (self.weight > other_fella.weight) and (self.height > other_fella.height):
            other_fella.rank+=1
        # elif (self.weight < other_fella.weight) and (self.height < other_fella.height):
        #     self.rank+=1
    def __str__(self):
        return f"{self.rank}"
"""덩치들 인스턴스 생성"""
# # N = int(input())
# Fellas_list = []
# for i in range(N):
#     weight, height = map(int, input().split())
#     Fellas_list.append(Fellas(weight, height))

# """대결시키기"""
# for i in range(len(Fellas_list)):
#     for j in range(len(Fellas_list)):
#         Fellas_list[i].is_bigger_than(Fellas_list[j])
# res_str=""
# for j in Fellas_list:
#     res_str+=f"{j.rank} "

# print(res_str)



#프로그래머스 DFS/BFS 타겟 넘버:
"""순열의 수 구하는 방법"""
# from itertools import permutations
# from itertools import combinations
# number_of_minus = 2
# num_list = [1,2,3,1]
# K = list(permutations(num_list, number_of_minus))
# # print(K)
# from itertools import permutations
# from itertools import combinations
# from collections import Counter
# from math import factorial
# def way_to_target(numbers, target):
#     number_of_minus = 1#마이너스로 바꿀 숫자의 수 
#     answer = 0#정답을 저장하는 변수
#     while number_of_minus <= len(numbers)-1: #마이너스가 5개가 되는 것은 의미가 없으므로(target이 1이상인 자연수임)
#         count = 0#해당 마이너스수에서 되는 조합이 몇개나 되는지 세려고---순열 값에 곱해줄것임
#         minus_numbers = list(combinations(numbers, number_of_minus))#마이너스수만큼 마이너스로 바꿀 수를 조합튜플로 리스트에 저장
#         for i in minus_numbers:#i가 마이너스로 바꿀 조합을 저장한 튜플임
#             temp = numbers  #temp초기화
#             for j in range(len(i)):#튜플의 길이에서-튜플의 각 값들에 접근하기 위함
#                 temp[temp.index(i[j])]= -1*i[j]#넘버스에서 뽑아낸 값들이므로 무조건 인덱스에 값이 있음/ 그 값에 마이너스를 곱한 수를 저장
#             new_sum = sum(temp) #해당 튜플에서의 작업이 끝나고, 전체합을 구한다
#             if new_sum == target:#전체합이 타깃과 같으면 가능한 조합이 하나 나온것
#                 count +=1
#         number_of_minus += 1
#         # answer += count*(len(list(permutations(numbers, number_of_minus))))
#         answer += factorial()
#     """중복되는 숫자가 있을 경우 --같은 것이 있는 순열--"""
#     count = Counter(i for i in temp)

#     return answer
# print(way_to_target([1,1,1,1,1], 3))
"""위 방식의 한계: 같은 수들이 있을때(예시와 같이) 하나하나 감안을 못해줌?-Counter로 하면 되잖아"""
# 같은 것이 있는 순열 형식으로 푸는것이 너무 복잡하고 비효율적인듯.



from collections import deque
def min_blank_max(s):
    arranged_queue=deque()
    s = s+' '
    queue = list(s)
    start =0
    for i in range(len(queue)):
        if queue[i]==' ':
            temp = queue[start:i]
            arranged_queue.append(int(''.join(temp)))
            start = i+1
    maximum = max(arranged_queue); minimum = min(arranged_queue)
    return f"{minimum} {maximum}"
# print(min_blank_max("4561 -562 783 -94"))
import copy
def one_two_four_zinbub(n):
    our_num = ''
    N = copy.deepcopy(n)
    # remainder = -1
    while N!=0:
        remainder = N%3
        N = N/3
        if remainder == 0:
            our_num += '4'
            N-=1
        elif remainder ==1:
            our_num += '1'
        elif remainder == 2:
            our_num += '2'
    our_num = our_num[::-1]
    return int(our_num)
# print(one_two_four_zinbub(15))


#프로그래머스 Lv2 소수찾기
from math import ceil
from itertools import permutations
def sosuchatgi(numbers):
    numbers = [int(i) for i in numbers]
    numbers.sort()
    numbers = [str(i) for i in numbers]
    integrated_num_list = []
    num_list=[]
    count = 0
    if numbers.count('0') == len(numbers):
        return count
    elif numbers.count(numbers[0]) == len(numbers):
        temp_boo = True
        for i in range(2,int(numbers[0])):
            if int(numbers[0])%i==0: 
                temp_boo = False
            if temp_boo == False:
                return count

    else:
        for i in range(1,len(numbers)+1):
            integrated_num_list.append(list(permutations(numbers, i)))
        for i in range(len(integrated_num_list)):
            for j in range(len(integrated_num_list[i])):
                K=''
                Boo = True
                for k in range(len(integrated_num_list[i][j])):
                    K += integrated_num_list[i][j][k]
                if K not in num_list:
                    num_list.append(K)
                elif K in num_list:
                    break
                K = int(K)
                if K//2>=2:
                    for h in range(2,ceil(K**0.5)):
                        if K%h == 0:
                            Boo = False
                if Boo and K!=1 and K != 4:
                    count+=1    
        return count
# print((sosuchatgi("17")))
# print(sosuchatgi("011"))
# print(sosuchatgi('7843'))
# print(sosuchatgi('99999'))
# # print(sosuchatgi('1276543'))
# # print(int('011'))
# zero= '2000'
# print(zero.count(zero[0])==len(zero))



#다음 큰 숫자
def count_one(N):
    binary_num = bin(N)
    counting_one = binary_num.count('1')
    return counting_one
    
def next_bigger_num(n):
    for num in range(n+1,1000000):
        if count_one(n)==count_one(num):
            return num
        break
# print(count_one(78))
# print(count_one(83))



#H-index
def Hindex(citations):
    citations.sort(reverse=True)
    h = max(citations)#h는 h의 가능한 최댓값인 전체 논문의 수로 시작함--1씩 빼면서 탐색
    while(True):
        over_cnt = 0
        for i in citations:
            if i>=h:
                over_cnt+=1 #over_cnt는 h번 이상 인용된 논문의 수를 뜻함
        if over_cnt >= h and len(citations) - over_cnt <= h:
            return h #조건 1: (h번 이상 인용된 논문의 수 가 h개 이상) 조건2: (h번 이하 인용된 논문의 수가 h개 이하)
        h-=1
    return 0
# print(Hindex([3,0,6,1,5]))


#더 맵게
#try1 -- 효율성 탈락(Nlg(N))
def spicier(scoville, K):
    count =0
    # scoville.sort(reverse=True)
    while min(scoville)<K:
        scoville.sort(reverse=True)
        food1 = scoville.pop()
        food2 = scoville.pop()
        new_food=food1+(food2*2)

        scoville.append(new_food)
        count+=1
        
        if len(scoville)<=1:
            return -1
    return count

# queue = deque([1,2,3])
# print(list(queue))

#가장 작은 수를 첫 번째 인덱스에 오게 하는 함수 reverse_heapifyv 
#2try -- 효율성탈락, 오답많음
# def swap(tree, index1, index2):
#     temp = tree[index1]
#     tree[index1] = tree[index2]
#     tree[index2] = temp

def reverse_heapify(tree, index):
    parent_index = index//2
    smaller = index
    if (parent_index>0) and (tree[index]<tree[parent_index]):
        smaller = parent_index
    if smaller != index:
        swap(tree, smaller ,index)
        reverse_heapify(tree, smaller)

# def extract_2mins(tree):
#     swap(tree, 1, -1)
#     min1 = tree.pop()
#     for i in range(1, len(tree)):
#             reverse_heapify(tree, i)
#     swap(tree, 1, -1)
#     min2 = tree.pop()
#     return [min1, min2]

def spicier2(scoville, K):
    scoville = deque(scoville)
    scoville.appendleft(None)
    scoville = list(scoville)
    count = 0

    # new_food = extract_2mins(scoville)

    while min(scoville[1:]) < K:#제일 매운 맛이 K가 될 때까지
        for i in range(1, len(scoville)):
            reverse_heapify(scoville, i)#제일 작은게 head가 되게함
        swap(scoville, 1, -1)#마지막이랑 위치를 바꾸고
        min1 = scoville.pop()#꺼낸다
        swap(scoville, 2, -1)#두 번쨰랑 위치를 바꾸고
        min2 = scoville.pop()#꺼낸다
        new_food = min1 + (min2)*2 #꺼낸 것들의 합이 새로운 스코빌
        scoville.append(new_food)
        count+=1

        if len(scoville) <=1 and min(scoville[1:]) < K:
            return -1
    return count

# print(spicier([1,2,3,9,10,12],7))
        
        


#3try
import heapq
def spicier3(scoville, K):
    count = 0

    while min(scoville)<K:
        if len(scoville) <= 1:
            return -1
        else:
            heapq.heapify(scoville)
            min1 = heapq.heappop(scoville)
            min2 = heapq.heappop(scoville)
            new_scoville = min1 + (min2)*2

            heapq.heappush(scoville, new_scoville)
            count+=1
    return count
# print(spicier3([1,2,3,9,10,12],7))

#가장 큰 수
def quadraple_and_indexing(number):
    temp_str = ""
    number = str(number)
    temp_str+=number*4
    temp_str = temp_str[:4]
    return int(temp_str)

def biggest_number(numbers):
    if numbers.count(0) == len(numbers):
        return 0
    for i in range(len(numbers)):
        # numbers[i] = (triple_and_indexing(numbers[i]), numbers[i])
        if 0 < numbers[i]<10:
            numbers[i] = (quadraple_and_indexing(numbers[i]), numbers[i])
        elif 10<=numbers[i]<100:
            numbers[i] = (quadraple_and_indexing(numbers[i]), numbers[i])
        elif 100<=numbers[i]<1000:
            # numbers[i] = (int(str(numbers[i])+str(numbers[i])[-1]), numbers[i])
            numbers[i] = (quadraple_and_indexing(numbers[i]), numbers[i])
        else:
            numbers[i] = (numbers[i], numbers[i])
    numbers.sort(key = lambda x : (-x[0], x[1]))
    res_str = ""
    for h in range(len(numbers)):
        res_str += str(numbers[h][1])
    return res_str
# print(biggest_number([6,10,2]))
# print(biggest_number([40,403]))
# print(biggest_number([10,101]))
# print(biggest_number([1,11,111,1111]))
# print(biggest_number([]))

#Jaden Case
def Jadencase(string):
    string = string.lower()
    res_str = ""
    Boo = True
    for i in range(len(string)-1):
        if Boo == True:
            res_str+=string[i]
        if Boo == False:
            Boo = True
        if string[i] == " " and string[i+1].isalpha: 
            res_str+=string[i+1].upper()
            Boo = False
            # i+=1
    res_str += string[-1]
        
    # jaden = " ".join(text_list)
    return res_str
# print(Jadencase("3people unFollowed me"))

# from itertools import permutations
# def N_and_M(N, M):
#     num_list = [x for x in range(1, N+1)]
#     arr = list(permutations(num_list, M))
#     res_str = ""
#     for i in arr:
#         for j in range(M):
#             res_str += str(i[j])
            
#             res_str += " "
#         res_str += "\n"
#     return res_str
# # N, M =map(int, input().split())
# # print(N_and_M(N,M))
# import heapq
# from collections import deque
# def emergency_boat4(people, limit):
#     sorted_people =[]
#     for i in people:
#         heapq.heappush(sorted_people, (-i,i))
#     #받은 리스트를 내림차순으로 정렬
#     boat = 0
    
#     while len(people)>0:
#         limit_now = people[0]
#            #정답인 필요한 구명보트의 개수를 저장하는 변수, 현재 타고 있는 사람들의 무게합 변수.
#         for i in range(len(people)):
#             if (limit_now + people[0]) <= limit:# 
#             limit_now += heapq.heappop(people)
#         else: 
#             boat +=1
#             limit_now = 0        
#     return boat


# # print(emergency_boat4([70,50,80,50],100))        
# print(emergency_boat4([70,80,50],100))
# import heapq
# def nearest(num_list, target):
#     diff_list =[]
#     for i in num_list:
#         diff_list.append((abs(target - i),i))
#     heapq.heapify(diff_list)
#     # if 0 in diff_list:
#     #     diff_list.remove(0)
    
#     return diff_list[0][1]

# import copy
# def emergency_boat2_2(people, limit):
#     people.sort(reverse = True)
#     boat = 0
#     iterator = copy.deepcopy(limit)
#     while len(people)>0:
#         if iterator - nearest(people, iterator) >= 0:
#             temp_iterator = copy.deepcopy(iterator)
#             nearest_num = nearest(people, iterator)
#             iterator -= nearest_num
#             people.remove(nearest_num)
#             break
#         boat +=1
#         iterator = copy.deepcopy(limit)
#     return boat
# --------------------2명 조건이 있는 이상 nearest 함수가 의미가 없어졌다
# print(emergency_boat2_2([70,50,80,50],100))        
# print(emergency_boat2_2([70,80,50],100))

def emergency_boat3(people, limit):
    boat =0
    
    people.sort(reverse= True)
    # people.append(people[-1])
    while len(people)>0:
        current_weight = people[0]
        del(people[0])
        for i in range(len(people)):
            
            if current_weight + people[i] <=limit:
                del(people[i])
                break

        boat+=1
    return boat
# print(emergency_boat3([70,50,80,50],100))        
# print(emergency_boat3([70,80,50],100))
# print(emergency_boat3([20,50,80,50],100))

#Yaksu
def yaksu_sum(N):
    sum =0
    for i in range(1,N+1):
        if N%i ==0:
            sum +=i
    return sum
# print(yaksu_sum(12))
# print(yaksu_sum(5))

def sum_between_two(a,b):
    sum =0
    if a<=b:
        for i in range(a,b+1):
            sum+=i
    else:
        for j in range(b, a+1):
            sum+=j
    return sum
# print(sum_between_two(3,5))
# print(sum_between_two(3,3))
# print(sum_between_two(5,3))

from math import gcd
def LCM2(arr):
    for i in range(len(arr)-1):
        GCD = gcd(arr[i], arr[i+1])

    LCM = GCD
    for i in range(len(arr)):
        arr[i] = arr[i]/GCD
        LCM *= arr[i]
    return int(LCM)

# print(LCM([2,6,8,14]))
# print(LCM([1,2,3]))

def n_tuple(s):
    new_list = []
    start = 0 
    end = 0
    s = s[1:-1]
    for i in range(len(s)):
        if s[i] == '{':
            temp =[]
        elif s[i] == '}':
            new_list.append(temp)
        
    # new_list.sort(key=len)
    # answer = []
    # for i in range(len(new_list)):
    #     for j in range(len(new_list[i])):
    #         if (new_list[i][j] not in answer) and new_list[i][j] != ',':
    #             answer.append(int(new_list[i][j]))
    return new_list
# print(n_tuple("{{2},{2,1},{2,1,3},{2,1,3,4}}"))
# print(n_tuple("{{20,111},{111}}"))

from collections import Counter
def n_tuple_2(s):
    gwalho_count = s.count('{')
    gwalho_count -=1

    s = s.replace('{','')
    s = s.replace('}','')
    num_start =0
    num_end =0
    num_list =[]
    for i in range(len(s)-1):
        if s[i]==',':
            num_end = i
            num_list.append(int(s[num_start:num_end]))
            num_start = i+1
    num_list.append(int(s[num_start:]))
    answer = []
    counted_num = Counter(num_list).most_common()
    for i in range(gwalho_count):
        answer.append(counted_num[i][0])
        
    return answer
# print(n_tuple_2("{{2},{2,1},{2,1,3},{2,1,3,4}}"))
# print(n_tuple_2("{{20,111},{111}}"))


