"""프로그래머스 Lv2 멀쩡한 사각형"""
def swap(my_list, a,b):
    temp = my_list[a]
    my_list[a]=my_list[b]
    my_list[b] = temp
from math import gcd

def fine_square(w,h):
    #"""무조건 옆으로 긴 상황인 것으로 하자"""
    if w <=1 or h<=1:
        return 0
    elif w<h:
        return fine_square(h,w)
    elif w==h:
        return w**2-w
    else:
        return w*h - ((w+h)-gcd(w,h))
global Boo

"""skill_tree"""
def available_skilltree(skill, skill_trees):
    count =0
    for skill_tree in skill_trees:
        
        Boo = True
        for i in range(len(skill)-1, 0, -1):
            if skill[i] in skill_tree:
                for j in range(i, 0, -1):
                    try:
                        if skill_tree.index(skill[j]) > skill_tree.index(skill[j-1]):
                            pass
                        else:
                            Boo = False
                            break
                    except:
                        Boo = False
                        break
                else:
                    break   
        if Boo == True:
            count+=1
    return count

# print(available_skilltree("CBD", ["BACDE", "CBADF", "AECB", "BDA"]))


"""Greedy Algorithmn 큰 수 만들기"""
def making_big_num(number, k):
    for _ in range(k):
        test_part = list(map(int, number[:k]))
        else_part = number[k:]

        test_part.pop(test_part.index(min(test_part)))
        test_part = "".join(map(str, test_part))
        number = test_part + else_part
    return number
    
#시간초과 + 오류도 생김
# print(making_big_num("1924", 2))

def making_big_num2(number, k):
    number = list(map(int, number))
    for _ in range(k):
        for i in range(len(number)-1):
            if number[i] < number[i+1]:
                number.remove(number[i])
                break
    number = "".join(map(str,number))
    return number

# print(making_big_num2("1924", 2))
# print(making_big_num2("999", 2))

#나아졌지만 오답 1개, 시간초과 1개씩 있음

def making_big_num3(number, k):
    # number = list(map(int, number))
    for _ in range(k):
        for i in range(len(number)-1):
            if int(number[i]) < int(number[i+1]):
                number = number.replace(number[i],'')
                break
    # number = "".join(map(str,number))
    return number
# print(making_big_num3("1924", 2))
# print(making_big_num3("999", 2))

#심지어 이게 그전거보다 효율성이 나쁘네? int함수나 replace함수가 되게 별로인거인듯?

def making_big_num4(number, k):
    number = list(map(int, number))
    for _ in range(k):
        for i in range(len(number)-1):
            if number[i] < number[i+1]:
                number = number[:i]+number[i+1:]
                break
    number = "".join(map(str,number))
    return number
# print(making_big_num4("1924", 2))
#k값과 number자릿수가 어어엄청 큰가봄 메소드를 안써도 안된다..


def making_big_num5(number, k):
    number = list(map(int, number))
    subcount =0
    for _ in range(k):
        for i in range(len(number)-1):
            if number[i] < number[i+1]:
                if number[i] == 9:
                    break
                else:
                    number.remove(number[i])
                    subcount+=1
                    break
    
    while subcount<k:
        number.pop()
        subcount+=1
    
    number = "".join(map(str,number))
    return number
# print(making_big_num5("999", 2))


"""소수 찾기"""    #NUMBER ONE BY FAR:
#소수임을 판별해주는 함수
def is_prime_number(N):
    for i in range(2, N):
        if N%i == 0:
            return False
    return True

#문자열의 앞 부분에 0이 있으면 무조건 없애주는 함수
def delete_pre_zeros(string):
    if string[0]=='0':
        try:
            while string[0] == '0':
                string = string[1:]
        except:
            return ''
    return string

#main함수
from itertools import permutations
def searching_prime_numbers(numbers):
    all_kinds_of_num_list=[] #온갖 조합(사실 순열)이 들어가는 리스트(근데 중복을 뺀)
    candidate_list = [] #온갖 조합이 들어가는데 중복하는 것도 있는 리스트
    for i in range(1,len(numbers)+1):
        candidate_list.append(list(permutations(numbers,i)))#모든 개수별 순열을 싹다 넣는다(단, 튜플형태임)
    for candidate in candidate_list:
        for i in range(len(candidate)):
            if candidate[i] not in all_kinds_of_num_list:# 그중에서 중복을 제거하여 all_kinds_of~에 넣는다
                all_kinds_of_num_list.append(candidate[i])
    
    unprocessed_strings = [] #아직 소수인지 아닌지 확인을 안해본 리스트
    for j in range(len(all_kinds_of_num_list)):#각 개수별 튜플에 접근
        candidate_num = ''
        for pre_num in all_kinds_of_num_list[j]:
            candidate_num += pre_num
        right_before_processing = delete_pre_zeros(candidate_num)
        if (right_before_processing != '') and (right_before_processing not in unprocessed_strings):
            unprocessed_strings.append(right_before_processing)
    #preprocessing
    if '1' in unprocessed_strings:
        unprocessed_strings.remove('1')

    count_prime_number =0
    for prime_number_candidate in unprocessed_strings:
        if is_prime_number(int(prime_number_candidate)):
            count_prime_number +=1

    return count_prime_number

# print(searching_prime_numbers('011'))

"""최솟값 만들기"""
import heapq as h
def making_min(A,B):
    A=A
    h.heapify(A)
    heap = []
    for b in B:
        h.heappush(heap,(-b,b))
    sum =0
    for i in range(len(A)):
        sum += A[i] * heap[i][1]
    return sum
# print(making_min([1,4,2],[5,4,4]))
#배운점 : heapify 함수는 sort() 시킨다기보단, heap성질을 갖게 하는 것임! 차이가 있다! 이를통해 sort() 하고싶으면 heapsort()함수를 새로 짜야함
#다만, 최소한 리스트 중 최솟값이 0번 인덱스로 오는 것까지는 맞음!--이게 주 의미임
"""막간을 이용한 heapsort"""
import heapq

def heap_sort(nums):
  heap = []
  for num in nums:
    heapq.heappush(heap, num)
  
  sorted_nums = []
  while heap:
    sorted_nums.append(heapq.heappop(heap))
  return sorted_nums
#이러한 성질을 이용하여 K번째 최댓값이나 최솟값을 효율적으로 구현할 수 있다!

# print(heap_sort([4, 1, 7, 3, 8, 5]))

def making_min2(A,B):
    A.sort()
    B.sort(reverse = True)
    sum =0
    for i in range(len(A)):
        sum += A[i]*B[i]
    return sum
#요게 정답

"""소수 '만들기'"""

def is_prime_number2(N):
    if N == 1:
        return False
    else:
        for i in range(2,N):
            if N%i ==0:
                return False
        return True

from itertools import combinations
def making_prime_number(nums):
    count = 0
    list_of_picked_nums = list(combinations(nums, 3))
    for j in range(len(list_of_picked_nums)):
        if is_prime_number2(sum(list_of_picked_nums[j])):
            count+=1
    return count
# print(making_prime_number([1,2,3,4]))
# print(making_prime_number([1,2,7,6,4]))



"""#쿼드압축 후 개수 세기"""

#2차원 배열 안의 원소가 모두 같은지를 Boolean으로 나타내어 주는 함수
def is_all_same(arr):
    iterator = arr[0][0]
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if iterator != arr[i][j]:
                return False
    return True
from math import floor, ceil
global count_of_zero, count_of_one
count_of_one =0
count_of_zero =0

def quadraple_compression(arr):
    global count_of_zero, count_of_one

    # count_of_zero = 0
    # count_of_one = 0
    if len(arr) == 1 and len(arr[0]) ==1:
        if arr[0][0] == 0:
            count_of_zero +=1
        elif arr[0][0] == 1:
            count_of_one +=1

    elif is_all_same(arr):
        if arr[0][0] == 0:
            count_of_zero +=1
        elif arr[0][0] == 1:
            count_of_one +=1
    else:
        N = len(arr)
        mid = ceil((N-1)/2)
        quadraple_compression([[arr[i][j] for j in range(mid)] for i in range(mid)])
        quadraple_compression([[arr[i][j] for j in range(mid)] for i in range(mid, N)])
        quadraple_compression([[arr[i][j] for j in range(mid , N)] for i in range(mid)])
        quadraple_compression([[arr[i][j] for j in range(mid, N)] for i in range(mid, N)])

def answer(arr):
    global count_of_zero, count_of_one
    quadraple_compression(arr)
    return [count_of_zero, count_of_one]

# print(answer([[1,1,0,0],[1,0,0,0],[1,0,0,1],[1,1,1,1]]))
# print(quadraple_compression([[1,1,0,0],[1,0,0,0],[1,0,0,1],[1,1,1,1]]))

#클래스로 접근해볼까..?
# from math import ceil
# class ZeroOne_TwoDimension():
#     def __init__(self, arr):
#         self.arr = arr
#         self.count_of_zero = 0
#         self.count_of_one =0
    
#     def is_all_same(self):
#         arr = self.arr
#         iterator = arr[0][0]
#         for i in range(len(arr)):
#             for j in range(len(arr[0])):
#                 if iterator != arr[i][j]:
#                     return False
#         return True

    # def quadraple_compression(self):
    #     arr = self.arr
    #     if len(arr) == 1 and len(arr[0]) ==1:
    #         if arr[0][0] == 0:
    #             self.count_of_zero +=1
    #         elif arr[0][0] == 1:
    #             self.count_of_one +=1

    #     elif self.is_all_same():
    #         if arr[0][0] == 0:
    #             self.count_of_zero +=1
    #         elif arr[0][0] == 1:
    #             self.count_of_one +=1
    #     else:
    #         N = len(arr)
    #         mid = ceil((N-1)/2)
    #         return ZeroOne_TwoDimension([[arr[i][j] for j in range(mid)] for i in range(mid)]).quadraple_compression(),ZeroOne_TwoDimension([[arr[i][j] for j in range(mid)] for i in range(mid,N)]).quadraple_compression(),ZeroOne_TwoDimension([[arr[i][j] for j in range(mid,N)] for i in range(mid)]).quadraple_compression(),ZeroOne_TwoDimension([[arr[i][j] for j in range(mid,N)] for i in range(mid,N)]).quadraple_compression()

            # quadraple_compression([[arr[i][j] for j in range(mid)] for i in range(mid)]),
            # quadraple_compression([[arr[i][j] for j in range(mid)] for i in range(mid, N)]),
            # quadraple_compression([[arr[i][j] for j in range(mid , N)] for i in range(mid)]),
            # quadraple_compression([[arr[i][j] for j in range(mid, N)] for i in range(mid, N)])

# def real_quadraple_compression(arr):
#     new_arr = ZeroOne_TwoDimension(arr)
#     new_arr.quadraple_compression()

#     return [new_arr.count_of_zero, new_arr.count_of_one]

# print(real_quadraple_compression([[1,1,0,0],[1,0,0,0],[1,0,0,1],[1,1,1,1]]))

"""
#Task 2 (optional, extra credits)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation  # we need this to create an animation with matplotlib
from IPython.display import HTML  # we need this to show the animation with a control panel 

fig = plt.figure(figsize=(4,4))
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.plot(0,0,'yo',markersize=16)

r_Earth = 1
r_mars = 2

Earth_orbit = np.linspace(0,2*np.pi, 100)
mars_orbit = np.linspace(0,2*np.pi, 100)

x_Earth = r_Earth*np.cos(Earth_orbit)
y_Earth = r_Earth*np.sin(Earth_orbit)
plt.plot(x_Earth, y_Earth, 'b',lw=1)


x_mars = r_mars*np.cos(mars_orbit)
y_mars = r_mars*np.sin(mars_orbit)
plt.plot(x_mars, y_mars, 'r',linestyle ='dashed', lw=1)


#[DIY1] (optional, extra credits) 
# Q: what is the syntax to plot a "red triangle" at x=3 and  y=5 ? 
# You can write the question and answer as a text cell and include it in your report.
# If you like, you can even add a new code cell, type your answer and show the figure output
# When adding the answer, please specify the DIY number clearly in the text cell or code cell appropriately.

#draw the Earth's orbit as the blue solid line (Background)



# connect x_orbit and y_orbit and generate a blue, solid-line circle "Earth's orbit" (background)
# Set linewidth=1 (or lw=1) 



# ***** create many frames to make an animation --> Earth will be rotating around the Sun following the blue circle *****
# set the number of frames in the animation to be 500 
# If you use Nframe=500, it would take some time (more than minutes) to generate the animation.

Nframes =  50
# Nframes = 500

# initialize
# create an empty class "Earth," to be filled by data points (x,y) along the Earth's orbit for making an animation
# set the marker style to be a blue filled-circle with a size 8


Earth, = plt.plot([],[],'bo',markersize=8)
# mars, = plt.plot([],[],color = 'red', marker = 'D',markersize = 8 )
mars, = plt.plot([],[],'rD',markersize = 8 )



# initiate Earth, and mars, 
def init():
    Earth.set_data([],[])
    mars.set_data([],[])
    return Earth, mars,

# define the function animate(i) by adding stuff to generate frames for Earth, and mars, appropriately
def animate(i):
    x_Earth =  r_Earth*np.cos(-i*rotation)# x= r*cos(theta) where i*rotation = theta_i --> theta_i+1 = theta_i + rotation
    y_Earth =  r_Earth*np.sin(-i*rotation)# y= r*sin(theta)
    Earth.set_data(x_Earth,y_Earth)

    x_mars =  r_mars*np.cos(i*rotation)# x= r*cos(theta) where i*rotation = theta_i --> theta_i+1 = theta_i + rotation
    y_mars =  r_mars*np.sin(i*rotation)# y= r*sin(theta)
    mars.set_data(-x_mars,y_mars)
    return Earth, mars,

# set ranges between [-2, 2] for x- and y-axes
plt.xlim(-2,2)
plt.ylim(-2,2)
# put the title "Mars, Earth, and Sun" fontsize=20
plt.title('Mars, Earth, and Sun',fontsize=20)

# Keep the axes frames
plt.axis('on')
#plt.axis('off')

# set the x- and y- axes scale to be the same
plt.axis('equal')

# makes an animation by repeatedly calling a function func
anim = animation.FuncAnimation(fig, animate, init_func = init, frames = Nframes, interval=10, blit=False)

HTML(anim.to_jshtml())"""

#위에껀 이지우 과제 도와준 Visualizing Universe

"""프로그래머스 Lv3! 베스트 앨범"""
class Songs():
    def __init__(self, genre, hit, serial_num):
        self.genre = genre
        self.hit = hit
        self.serial_num = serial_num
        self.parent = None
        self.child = None

    def ranking(self, other_song):
        if self.genre == other_song.genre:
            if self.hit > other_song.hit:
                self.child = other_song
                self.child.parent = self
                
            elif self.hit == other_song.hit:
                if self.serial_num < other_song.serial_num:
                    self.child = other_song
                else:
                    self.parent = other_song
    
    def free_from_ranking(self):
        if self.child is not None:
            self.child.parent = None
        self.child = None

def making_best_album(genres, play):
    answer = []
    songs_list = []
    for i in range(len(genres)):
        songs_list.append(Songs(genres[i], play[i], i))
    #각기의 정보를 가진 곡 Songs 클래스 인스턴스 생성
    songs_list.sort(key= lambda x: x.hit, reverse= True)
    for j in range(len(songs_list)-1):
        songs_list[j].ranking(songs_list[j+1])

    #장르별 정보를 담을 딕셔너리, 장르 중복을 막기위한 집합화
    kinds_of_genres = set(genres)
    count_genres_called = {}
    all_hits_of_genre = {}

    #가장 조회수가 높은 장르 얻음, 장르가 2번 초과 호출되지 않도록 장르 호출 횟수를 담는 딕셔너리 생성
    for genre in kinds_of_genres:
        count_genres_called[genre] =0
        all_hits_of_genre[genre]=0
        for song in songs_list:
            if song.genre == genre:
                all_hits_of_genre[genre] += song.hit
    
    #조회수가 높은 장르별로 sort
    all_hits_of_genre = sorted(zip(all_hits_of_genre.keys(), all_hits_of_genre.values()),key = lambda x: x[1], reverse = True)
    
    #조회수가 높은 장르부터, 노래의 장르가 현재 검색중인  장르와 같다면, 해당 장르에서 조회수가 최고일것!--parent-child 시스템으로 정해보자
    
    for i in range(len(all_hits_of_genre)):
        for song in songs_list:
            if song.genre == all_hits_of_genre[i][0]:
                if song.parent == None:
                    if count_genres_called[song.genre] <2:
                        answer.append(song.serial_num)
                        count_genres_called[song.genre] = count_genres_called[song.genre] +1
                        song.free_from_ranking()

    

    return answer, [song.hit for song in songs_list], all_hits_of_genre #체크하기 위해--tip 이러면 함숫값이 튜플로 나온다!
# print(making_best_album(['classic', 'pop', 'classic', 'classic', 'pop'],[500, 600, 150, 800, 2500]))
# print(making_best_album(['classic','classic','classic','classic','pop'],[500,150,800,800,2500]))
# print(making_best_album(['A', 'A', 'B', 'A'], [5, 5, 6, 5]))

"""프로그래머스 Lv3! 네트워크 구현하기"""
#dfs를 하기 위해 adjacent 등의 관계성을 저장하는 특징을 가지고 있고, visited = False인 클래스 Computers를 생성

class Computers:
    def __init__(self, name):
        self.name = name
        self.adjacent_computers = []
        self.visited = False
    
    #연결을 보다 빠르게 해주는 함수
    def connecting_computers(self, other_computer):
        self.adjacent_computers.append(other_computer)
        other_computer.adjacent_computers.append(self)

from collections import deque
#클래스 인스턴스 생성과, 그래프 관계를 형상화시키는 함수
def show_connection(n, computer):
    computer_list = []
    for i in range(n):
        computer_list.append(Computers(i))
    for i in range(len(computer)):
        for j in range(len(computer[0])):
            if i != j:
                if computer[i][j] == 1:
                    if computer_list[j] not in computer_list[i].adjacent_computers:
                        computer_list[i].connecting_computers(computer_list[j])
    #bfs구현화
    queue = deque()
    count=0

    for computer in computer_list:
        computer.visited = False
    while len(computer_list)>0:
        computer_list[0].visited = True
        queue.append(computer_list.pop(0))
        while len(queue)>0:
            iterator = queue.popleft()
            for computer in iterator.adjacent_computers:
                if computer.visited == False:
                    computer.visited = True
                    computer_list.remove(computer)
                    queue.append(computer)
        count+=1
    return count
# print(show_connection(3,[[1, 1, 0], [1, 1, 0], [0, 0, 1]]))
# print(show_connection(3,[[1, 1, 0], [1, 1, 1], [0, 1, 1]]))
  
"""프로그래머스 Lv3! 단어 변환"""

#반복문과 조건문을 통한 검열
# def similarity(word1, word2):
#     count =0
#     for i in range(len(word1)):
#         if word1[i]==word2[i]:
#             count+=1
#     return count
 
def string_transformation(begin, target, words):
    count_try =0
    while similarity(begin, target) < len(begin):
        while len(words)>0:
            for word in words:
                if similarity(word, begin)==1:
                    if similarity(word, target) > similarity(begin, target):
                        begin = word
                        words.remove(word)
                        count_try +=1
            if min([similarity(begin, k) for k in words]) >1:
                return 0
    return count_try
# print(string_transformation('hit', 'cog',['hot', 'dot', 'dog', 'lot', 'log', 'cog']	))
#이러면 '최솟값' 이 아닐수도 있음!

#최단경로 알고리즘- BFS와 backtracking 활용

def similarity(word1, word2):
    count =0
    for i in range(len(word1.word)):
        if word1.word[i]!=word2.word[i]:
            count+=1
    return count

class Words:
    def __init__(self, word):
        self.word = word
        self.visited = False
        self.predecessor = None
        self.adjacent_words = []

    def add_to_adjacent_words(self, other_Word):
        self.adjacent_words.append(other_Word)
        other_Word.adjacent_words.append(self)

from collections import deque
def word_transformation2(begin, target, words):
    if target not in words:
        return 0
        #애초에 변형을 할 수가 없는 경우

    #클래스 인스턴스들을 생성하여 저장할 리스트
    Words_list =[]
    Words_list.append(Words(begin)); Words_list.append(Words(target))
    for i in range(len(words)):
        Words_list.append(Words(words[i]))
    
    #그래프 구조 형성
    for i in range(len(Words_list)-1):
        for j in range(len(Words_list)):
            if similarity(Words_list[i], Words_list[j])==1:
                if Words_list[j] not in Words_list[i].adjacent_words:
                    Words_list[i].add_to_adjacent_words(Words_list[j])
        
    #BFS preprocessing
    queue = deque()
    Words_list[0].visited = True
    queue.append(Words_list[0])

    #BFS 알고리즘과 predecessor를 적용한 backtracking 준비
    while (len(queue))>0:
        iterator = queue.popleft()
        for adjacent_word in iterator.adjacent_words:
            if adjacent_word.visited == False:
                adjacent_word.visited = True
                queue.append(adjacent_word)
                adjacent_word.predecessor = iterator
    
    backtracker = Words_list[1]
    count_try =0
    while backtracker.predecessor is not None:
        count_try +=1
        backtracker = backtracker.predecessor
        if count_try > len(Words_list):
            return 0
    if backtracker.word == begin:
        return count_try
    else:
        return 0
    # return count_try

# print(word_transformation2('hit', 'cog',['hot', 'dot', 'dog', 'lot', 'log', 'cog']	))

"""프로그래머스 Lv3! 여행 경로"""

# print('a'<'b') #CF: Ascii 코드 이용한 Boolean?

#출발지와 목적지 정보를 모두 갖고 있는 클래스
class Tickets:
    def __init__(self, ticket):
        self.fr = ticket[0]
        self.to = ticket[1]
        self.visited = False
        self.used = False
        self.adjacent_tickets = []
        self.descendent = None
    #한쪽 방향으로만 갈 수 있으므로 양쪽에 넣어줄 수는 없다
    def add_to_adjacent_tickets(self, other_ticket):
        self.adjacent_tickets.append(other_ticket)


from collections import deque
def travel_course(tickets):
    Tickets_list =[]
    starting_Tickets_list = []
    #클래스 인스턴스 생성과, 시작점은 따로 분류
    for i in range(len(tickets)):
        Tickets_list.append(Tickets(tickets[i]))
    for Ticket in Tickets_list:
        if Ticket.fr =='ICN':
            starting_Tickets_list.append(Ticket)

    #그래프 연결구조 생성--목적지와 출발지가 같을 때 --거꾸로도 한번 탐색해준다
    for i in range(len(Tickets_list)-1):
        for j in range(1,len(Tickets_list)):
            if Tickets_list[i].to == Tickets_list[j].fr:
                Tickets_list[i].add_to_adjacent_tickets(Tickets_list[j])
            elif Tickets_list[j].to == Tickets_list[i].fr:
                Tickets_list[j].add_to_adjacent_tickets(Tickets_list[i])
    
    #시작 리스트를 정렬해서 순서대로 들어가도록
    queue=deque()
    starting_Tickets_list.sort(key = lambda x: x.to)
    #각 시작리스트별 탐색한 경로를 담아주고, 그중에 abc순서대로+모든 티켓을 소진한 것. 을 리턴하자
    answer_candidates = []
    
    #시작이 가능한 노드부터 시작할것임
    while len(starting_Tickets_list)>0:
        possible_routes = []
        starting_Tickets_list[0].visited = True
        head_node = starting_Tickets_list[0]
        queue.append(starting_Tickets_list.pop(0))
    
        while len(queue)>0:
            iterator = queue.pop()
            iterator.adjacent_tickets.sort(key = lambda x : x.to)

            for next_ticket in iterator.adjacent_tickets:
                if next_ticket.visited == False:
                    next_ticket.visited = True
                    queue.append(next_ticket)
                    iterator.descendent = next_ticket
                    break
    
        back_backtracker = head_node
        # tracking_count=0
        while back_backtracker.descendent is not None:
            possible_routes.append(back_backtracker.fr)
            back_backtracker = back_backtracker.descendent
        possible_routes.append(back_backtracker.fr)
        possible_routes.append(back_backtracker.to)
        # back_backtracker = back_backtracker.descendent
        
        answer_candidates.append(possible_routes)
    answer_candidates.sort()
    # return answer_candidates
    for answer in answer_candidates:
        if len(answer) == (len(tickets)*2) -(len(tickets)-1):
            return answer
        else:
            answer.pop()
            while len(answer)<(len(tickets)*2) -(len(tickets)-1)-1:
                for others in Tickets_list:
                    if others.used == False:
                        if back_backtracker.to == others.fr:
                            others.used = True
                            answer.append(others.fr)
                            back_backtracker = others
            answer.append(back_backtracker.to)
            return answer            


# print(travel_course([['ICN', 'SFO'], ['ICN', 'ATL'], ['SFO', 'ATL'], ['ATL', 'ICN'], ['ATL','SFO']]))
# print(travel_course([['ICN', 'A'], ['ICN', 'B'], ['B', 'ICN']]))


# class Tickets:
#     def __init__(self, ticket):
#         self.fr = ticket[0]
#         self.to = ticket[1]
#         self.visited = False
#         self.used = False
#         self.adjacent_tickets = []
#         self.descendent = []
#     #한쪽 방향으로만 갈 수 있으므로 양쪽에 넣어줄 수는 없다
#     def add_to_adjacent_tickets(self, other_ticket):
#         self.adjacent_tickets.append(other_ticket)


# from collections import deque
# def travel_course(tickets):
#     Tickets_list =[]
#     starting_Tickets_list = []
#     #클래스 인스턴스 생성과, 시작점은 따로 분류
#     for i in range(len(tickets)):
#         Tickets_list.append(Tickets(tickets[i]))
#     for Ticket in Tickets_list:
#         if Ticket.fr =='ICN':
#             starting_Tickets_list.append(Ticket)

#     #그래프 연결구조 생성--목적지와 출발지가 같을 때 --거꾸로도 한번 탐색해준다
#     for i in range(len(Tickets_list)-1):
#         for j in range(1,len(Tickets_list)):
#             if Tickets_list[i].to == Tickets_list[j].fr:
#                 Tickets_list[i].add_to_adjacent_tickets(Tickets_list[j])
#             elif Tickets_list[j].to == Tickets_list[i].fr:
#                 Tickets_list[j].add_to_adjacent_tickets(Tickets_list[i])
    
#     #시작 리스트를 정렬해서 순서대로 들어가도록
#     queue=deque()
#     starting_Tickets_list.sort(key = lambda x: x.to)
#     #각 시작리스트별 탐색한 경로를 담아주고, 그중에 abc순서대로+모든 티켓을 소진한 것. 을 리턴하자
#     answer_candidates = []
    
#     #시작이 가능한 노드부터 시작할것임
#     while len(starting_Tickets_list)>0:
#         possible_routes = []
#         starting_Tickets_list[0].visited = True
#         head_node = starting_Tickets_list[0]
#         queue.append(starting_Tickets_list.pop(0))
    
#         while len(queue)>0:
#             iterator = queue.pop()
#             iterator.adjacent_tickets.sort(key = lambda x : x.to)

#             for next_ticket in iterator.adjacent_tickets:
#                 if next_ticket.visited == False:
#                     next_ticket.visited = True
#                     queue.append(next_ticket)
#                     iterator.descendent.append(next_ticket)
                    
    
#         back_backtracker = head_node
#         # tracking_count=0
#         while back_backtracker.descendent is not None
#             if back_backtracker.used is False:
#                 possible_routes.append(back_backtracker.fr)
#                 back_backtracker.used = True
#                 back_backtracker = back_backtracker.descendent
#         possible_routes.append(back_backtracker.fr)
#         possible_routes.append(back_backtracker.to)
        
#         answer_candidates.append(possible_routes)
#     answer_candidates.sort()
#     return answer_candidates
#     # for answer in answer_candidates:
#     #     if len(answer) == (len(tickets)*2) -(len(tickets)-1):
#     #         return answer

# print(travel_course([['ICN', 'SFO'], ['ICN', 'ATL'], ['SFO', 'ATL'], ['ATL', 'ICN'], ['ATL','SFO']]))
# print(travel_course([['ICN', 'A'], ['ICN', 'B'], ['B', 'ICN']])) 


"""타겟 넘버"""
from itertools import combinations
import copy
def target_number(numbers, target):
    minus_num_tuple_list =[]
    count =0
    temp=copy.deepcopy(numbers)
    # temp = numbers
    for i in range(1, len(numbers)):
        minus_num_tuple_list.append(list(combinations(numbers,i)))
    for i in range(len(minus_num_tuple_list)):
        for minus_tuple in minus_num_tuple_list[i]:
            for minus_num in minus_tuple:
                temp.remove(minus_num)
                temp.append(-1*minus_num)
            if sum(temp) == target:
                count+=1
            temp = copy.deepcopy(numbers)
    return count
# print(target_number([1,1,1,1,1],3))

"""(복습) Jaden Case"""
def JadenCase(string):
    string = string.lower()
    res_str = ""
    Boo = True
    for i in range(len(string)-1):
        if i ==0 and string[i].isalpha():
                res_str+=string[i].upper()
        else:
            if Boo == True:
                res_str+=string[i]
            if Boo == False:
                Boo = True
            if string[i] == " " and string[i+1].isalpha: 
                res_str+=string[i+1].upper()
                Boo = False
    res_str += string[-1]
    return res_str


"""조이스틱""" #--------------------------------------------------------못하겠어! 못하겠다고!!! ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
# #global 변수 MOVE
# global TOTAL_MOVE

# import copy
# class Cursor:
#     def __init__(self, number):
#         self.number = number
#         self.status = 65 #Ascii 코드로 A--chr(65) == 'A' +=1씩 하면 B,C,D.. chr(90)=='Z'
#         self.left = None
#         self.right = None
    
#     def plus_alphabet(self):
#         if self.status<90:
#             self.status+=1
#         else:
#             self.status = 65

#     def minus_alphabet(self):
#         if self.status>65:
#             self.status -=1
#         else:
#             self.status = 65
    
#     #target 은 str인 알파벳 한  철자.
#     #더 짧은 루트를 계산하여 리턴함
#     def change_letter_in_best_route(self, target):
#         plus_cand = copy.deepcopy(self)
#         minus_cand = copy.deepcopy(self)
#         plus_count =0
#         minus_count =0
#         while plus_cand.status != ord(target):
#             plus_cand.plus_alphabet()
#             plus_count+=1
#         while minus_cand.status != ord(target):
#             minus_cand.minus_alphabet()
#             minus_count +=1
#         return min(minus_count, plus_count)

#     def move_cursor_in_best_route(self, other_cursor):
#         plus_route = copy.deepcopy(self)
#         minus_route = copy.deepcopy(self)
#         plus_count_cursor =0
#         minus_count_cursor =0

#         while plus_route.number != other_cursor.number:
#             plus_route = plus_route.right
#             plus_count_cursor +=1
#         while minus_route.number != other_cursor.number:
#             minus_route = minus_route.left
#             minus_count_cursor+=1
#         return min(plus_count_cursor, minus_count_cursor)
    
# def Joystick(name):
#     global TOTAL_MOVE
#     TOTAL_MOVE=0
#     #커서 생성
#     cursor_list=[]
#     for i in range(len(name)):
#         cursor_list.append(Cursor(i))
#     #변환 시작
#     current_cursor = cursor_list[0]
    
#     for cursor in cursor_list:
#         res_str = ""
#         res_str += chr(cursor.status)
#         if res_str != name:
#             phase =current_cursor.number
#             if current_cursor.status != ord(name[phase]):
#                 TOTAL_MOVE+=current_cursor.change_letter_in_best_route(name[phase])
#                 current_cursor.status = ord(name[phase])
#                 current_cursor=current_cursor.right()

#             else:
#                 current_cursor=current_cursor.left()


"""2019 카카오 개발자 인턴십--인형뽑기"""
def Kakao_doll_machine(board, move):
    stack = [0]
    count =0
    for column in move:
        for length_of_square in range(len(board)):
            if board[length_of_square][column-1] != 0:
                temp = board[length_of_square][column-1]
                board[length_of_square][column-1]=0
                stack.append(temp)
                if stack[-1] == stack[-2]:
                    count +=2
                    stack.pop()
                    stack.pop()
                break
    return count
# print(Kakao_doll_machine([[0,0,0,0,0],[0,0,1,0,3],[0,2,5,0,1],[4,2,4,4,2],[3,5,1,3,1]],[1,5,3,5,1,2,1,4]))

"""완주하지 못한 선수"""
class Participants:
    def __init__(self, name):
        self.name = name
        self.participant = False
        self.completion = False
        
def loser1(participant, completion):
    participants_list =[]
    for i in range(len(participant)):
        participants_list.append(Participants(participant[i]))
        
    for Participant in participants_list:
        if Participant.name in participant:
            Participant.participant = True
        if Participant.name in completion:
            Participant.completion = True
            completion.remove(Participant.name)
    for loser in participants_list:
        if (loser.participant is True) and (loser.completion is False):
            return loser.name
# print(loser1(['leo', 'kiki', 'eden'],['eden', 'kiki']))
# print(loser1(['mislav', 'stanko', 'mislav', 'ana'],['stanko', 'ana', 'mislav']))

#npsetdiff1d 가 교집합을 전부 빼버려서 안됨
import numpy as np
def loser2(participant, completion):
    arr1 = np.array(participant)
    arr2 = np.array(completion)
    answer = np.setdiff1d(arr1, arr2)
    return answer[0]

import copy
def loser3(participants, completion):
    copy_list = copy.deepcopy(participants)
    for participant in copy_list:
        if participant in completion:
            participants.remove(participant)
            completion.remove(participant)
    return participants[0]

def loser4(participant, completion):
    hash ={}
    for people in participant:
        if people in hash:
            hash[people] +=1
        else:
            hash[people] =1
    for winner in completion:
        if winner in completion:
            hash[winner] -=1
    for people in participant:
        if hash[people]>0:
            return people


# print(loser4(['leo', 'kiki', 'eden'],['eden', 'kiki']))
# print(loser4(['mislav', 'stanko', 'mislav', 'ana'],['stanko', 'ana', 'mislav']))
