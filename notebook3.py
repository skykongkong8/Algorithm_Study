"""완주하지 못한 선수--해시"""

#해시는 특정한 정보를 다른 무작위의 수로 치환해서 저장하는 방식이다
def loser_final1(participant, completion):
    answer = ''
    hash = {}
    for name in completion :
        if name in hash :
            hash[name] += 1
        else:
            hash[name] = 1

    for key in participant :
        if key in hash :
            hash[key] -= 1
            if hash[key] < 0:
                answer = key
        else :
            answer = key

    return answer

#Counter 함수 이용하기 -- Counter는 dict형태로 각각의 내용들을 세어준다. {key=항목: value=개수}
import collections
def loser_final2(participant, completion):
    answer = collections.Counter(participant) - collections.Counter(completion)
    return list(answer.keys())[0]
#dict끼리의 연산이 되는 것이 아니다, Counter는 dict의 형태일 뿐 dict인 것은 아니다. Counter 끼리의 연산이 되는 것이다.
#이때 그 연산은 집합의 차집합처럼 이뤄지는데, 그래도 중복된 것의 수를 지킬 수가 있게된다(np.array집합연산은 안되는데 반해)--중복된 것은 이미 개수로 병합되었으므로.

"""모의고사"""
class Student:
    def __init__(self, way_of_picking, correct, number):
        self.way_of_picking = way_of_picking
        self.correct = correct
        self.number = number

def model_test(answer):
    student1 = Student([1,2,3,4,5]*2000,0,1) #5
    student2 = Student([2,1,2,3,2,4,2,5]*1250,0,2) #8
    student3 = Student([3,3,1,1,2,2,4,4,5,5]*1000,0,3) #10

    student_list =[student1, student2, student3]
    max_student = [student1]
    for student in student_list:
        for n in range(len(answer)):
            if answer[n] == student.way_of_picking[n]:
                student.correct+=1
        if student.correct>max_student[0].correct:
            max_student.clear()
            max_student.append(student)
        elif student.correct == max_student[0].correct and student.number != max_student[0].number:
            max_student.append(student)
    answer =[]
    for estudiante in max_student:
        answer.append(estudiante.number)      
    return answer
# print(model_test([1,2,3,4,5]))
# print(model_test([1,3,2,4,2]))

"""2016년"""
# 윤년에서 요일 특정하기 CF) 윤년 = 2월이 29일까지만 있는 해.
#31 29 31 30 31 30 31 31 30 31 30 31일(각 월당)
# 2016년 1월 1일은 금요일임.
def dosmilesdieciseisanos(a,b):
    days_list = [31,29,31,30,31,30,31,31,30,31,30,31]
    days_passed=0
    for i in range(a-1):
        days_passed+=days_list[i]
    days_passed += b
    if days_passed % 7 == 1:
        return "FRI"
    elif days_passed % 7 == 2:
        return "SAT"
    elif days_passed % 7 == 3:
        return "SUN"
    elif days_passed % 7 == 4:
        return "MON"
    elif days_passed % 7 == 5:
        return "TUE"
    elif days_passed % 7 == 6:
        return "WED"
    elif days_passed % 7 == 0:
        return "THU"
# print(dosmilesdieciseisanos(5,24))

"""가운데 글자 가져오기"""
def bringing_middle_letter(s):
    if len(s)%2 == 1: #홀수이면
        return s[len(s)//2]
    elif len(s)%2 == 0: #짝수이면
        return s[len(s)//2-1:(len(s)//2)+1]

# print(bringing_middle_letter('qwer'))    


"""월간 코드 챌린지--3진법 뒤집기"""
def swifting_samjinbub(n):
    #10진법 수를 3진법으로 바꾸기:
    samjinbub_num =0
    res_str=""
    while 3**samjinbub_num <= n:
        samjinbub_num+=1
    
    for iterator in range(samjinbub_num, -1,-1):
        count = n//3**(iterator-1)
        n %= 3**(iterator-1)
        
        res_str+=str(count)
        if n==0:
            break
        samjinbub_num = 0
        
        for _ in range(samjinbub_num-1):
            res_str+='0'

    # #제작한 3진법 수를 뒤집기
    # res_str= res_str[::-1]
    
    #뒤집은 3진법 수를 10진법으로 바꾸기
    answer =0
    power = 0
    for num in res_str:
        this_number = int(num) * 3 ** power
        answer += this_number
        power +=1
    
    return answer

# print(swifting_samjinbub(45))
# print(swifting_samjinbub(125))
# print(swifting_samjinbub(1))
# print(swifting_samjinbub(78413450))

"""문자열 sorting lambda x 자유자재로 다루기"""
#1 다중 조건은 튜플로 하면 되지만, 까다로울 경우 두 번 해서 유지시키는 경우
def i_am_string_sorting_master(strings, n):
    strings.sort()
    strings.sort(key = lambda x: x[n])
    return strings
# print(i_am_string_sorting_master(['abce', 'abcd', 'cdx'],2))

#2 문자열 그 자체는 sort 메소드가 없기 때문에 리스트로 바꿨다가 join 하는 경우
def i_am_string_sorting_master2(string):
    string = list(string)
    string.sort(key = lambda x: ord(x), reverse = True)
    string = "".join(string)
    return string
# print(i_am_string_sorting_master2('Zbcdefg'))


"""소수 찾기"""

#1 절반의 범위 내에서만 탐색해도 알 수 있다는 가정
def finding_prime_num(n):
    count =0
    for N in range(2, n+1):
        if N%2 ==0:
            break
        for i in range(2, (N//2)+1):
            if N%i == 0:
                break
        else:
            count+=1
    return count
# print(finding_prime_num(10))
# print(finding_prime_num(5))

#2 에라토스테네스의 체--Boolean으로 공배수가 중복되서 지워지는 현상을 방지할 수 있다!
from math import sqrt, ceil

def eratostenes(n):
    is_primes = [True]*(n+1) #자기자신까지 포함해야 하는 경우. 아니면 n까지만 하면 된다
    max_length = ceil(sqrt(n)) # n = ab라고 했을때, 무조건 sqrt(n) 이하일 것이므로
    for i in range(2,max_length):
        if is_primes[i]:
            for j in range(i+i, n+1, i): #i를 빼고, i의 다음 배수부터, 전체 길이까지, i씩 짚어가면서,
                is_primes[j] = False
    return len([i for i in range(2,n+1) if is_primes[i]]) #Boolean을 사용하는 방법.
# print(get_prime(5))
# print(get_prime(10)) 

"""시저 암호문"""
#Ascii 코드를 사용해야 함
class Letters:
    def __init__(self, letter):
        self.letter = letter
        self.ascii_num = ord(letter)
    
    def move_ascii_num(self):
        #소문자
        if (65<= self.ascii_num < 90) or (97<=self.ascii_num<122):
            self.ascii_num +=1
        elif self.ascii_num == 122:
            self.ascii_num = 97
        elif self.ascii_num == 90:
            self.ascii_num = 65 

        self.letter = chr(self.ascii_num)

def caesar_code(s,n):
    res_str = ""
    Letters_list = []
    for i in range(len(s)):
        Letters_list.append(Letters(s[i]))
    for Letter in Letters_list:
        for _ in range(n):
            Letter.move_ascii_num()
        res_str += Letter.letter
    return res_str


"""내적 구현하기"""
#numpy를 쓰자
"""numpy 쓸 때 유의할 점: TypeError: Object of type int64 is not JSON serializable\
    예시) 어떤 리스트를 처리하기 위해 np.array로 바꾸었다 치자. 그러면 해당 리스트 내 원소 변수 타입은 numpy.int64가 된다\
    그런데 이후 다시 결과값을 도출하기 위해여 list로 변환을 하면, 여전히 내부 원소 변수 타입은 numpy.int64이기 때문에
    json을 활용하여 처리할 경우, 이를 인식하지 못하여 에러가 발생한다. 그것이 위의 경우인 것.
    따라서, 이 때는(list를 np.array로 바꾸었고, json을 사용할 경우) 내부 원소까지 모두 int형으로 바꿔 준 후에 진행해야 한다!"""
# 예시) --visual studio code에 numpy가 안깔림,,ㅎ
# import numpy as np
# def solution(a, b):
#     a = np.array(a)
#     b = np.array(b)
    
#     answer = int(np.dot(a,b))
#     return answer

"""이상한 문자 만들기"""
class Letters2:
    def __init__(self, letter, index):
        self.letter = letter
        self.ascii_num = ord(letter)
        self.index = index
    
    def switch_ascii_num(self):
        #소문자
        if (65<= self.ascii_num <= 90) or (97<=self.ascii_num<=122):
            if self.index%2 != 0:
                if (65<= self.ascii_num <= 90):
                    self.ascii_num +=32
                elif (97<=self.ascii_num<=122):
                    # self.ascii_num -= 32
                    pass
            if self.index%2 == 0:
                if (65<= self.ascii_num <= 90):
                    # self.ascii_num +=32
                    pass
                elif (97<=self.ascii_num<=122):
                    self.ascii_num -= 32
                    # pass

        self.letter = chr(self.ascii_num)

def weird_string(s):
    split_letters = s.split(' ')
    answer =[]
    for word in split_letters:
        res_str = ""
        Letters2_list = []
        for i in range(len(word)):
            Letters2_list.append(Letters2(word[i],i))
        for Letter2 in Letters2_list:
            Letter2.switch_ascii_num()
            res_str += Letter2.letter
        answer.append(res_str)
    return " ".join(answer)
# print(weird_string('try hello world'))


"""2020 카카오 인턴십 -- 키패드 누르기"""
class Hands:
    def __init__(self, direction, status):
        self.direction = direction
        self.status = status #자신의 현 위치는 튜플로 나타내도록 하자?
    
    def distance_from_button(self, num, keypad):
        for i in range(len(keypad)):
            for j in range(len(keypad[0])):
                try:
                    if keypad[i][j].number == num:
                        distance = abs(self.status[0] - i) + abs(self.status[1] - j)
                except:
                    pass
        return distance
    
    def move_to_button(self, num, keypad):
        for i in range(len(keypad)):
            for j in range(len(keypad[0])):
                try:
                    if keypad[i][j].number == num:
                        self.status = (i,j)
                except:
                    pass

class KeyPadButton:
    def __init__(self, number):
        self.number = number

def keypad_pushing(numbers, hand):
    res_str = ""
    #손 객체 생성
    Right_Hand = Hands('R', (3,0))
    Left_Hand = Hands('L', (3,2))

    #키패드 객체 생성-인덱스가 곧 정보
    KeyPad =[[KeyPadButton(1), KeyPadButton(2), KeyPadButton(3)],
    [KeyPadButton(4),KeyPadButton(5),KeyPadButton(6)],
    [KeyPadButton(7),KeyPadButton(8),KeyPadButton(9)],
    [KeyPadButton('*'),KeyPadButton(0),KeyPadButton('#')]]
    
    #버튼 누르기
    for number in numbers:
        if number in [1,4,7]:
            Left_Hand.move_to_button(number, KeyPad)
            res_str+='L'
        elif number in [3,6,9]:
            Right_Hand.move_to_button(number, KeyPad)
            res_str += 'R'
        elif number in [2,5,8,0]:
            left_hand_distance = Left_Hand.distance_from_button(number, KeyPad)
            right_hand_distance = Right_Hand.distance_from_button(number, KeyPad)

            if left_hand_distance < right_hand_distance:
                Left_Hand.move_to_button(number, KeyPad)
                res_str+='L'
            elif left_hand_distance > right_hand_distance:
                Right_Hand.move_to_button(number, KeyPad)
                res_str += 'R'
            elif left_hand_distance == right_hand_distance:
                if hand == 'left':
                    Left_Hand.move_to_button(number, KeyPad)
                    res_str += 'L'
                elif hand == 'right':
                    Right_Hand.move_to_button(number, KeyPad)
                    res_str += 'R'

    return res_str
# print(keypad_pushing([1, 3, 4, 5, 8, 2, 1, 4, 5, 9, 5], 'right'))

"""GCD and LCM"""
#1 내장 모듈 사용
from math import gcd
def GCDandLCM(n,m):
    GCD = gcd(n,m)
    lcm = GCD*(n/GCD)*(m/GCD)
    return [GCD, int(lcm)]
# print(GCDandLCM(3,12))

#2 유클리디언 알고리즘 활용
def gcdlcm(a, b):
    c, d = max(a, b), min(a, b) #두 수 중 더 큰 수를 찾는다.
    t = 1
    while t > 0:
        t = c % d #t는 c를 d로 나눈 나머지
        c, d = d, t #c를 d로 업데이트, d를 t로 업데이트
    answer = [c, int(a*b/c)]

    return answer
    #솔직히 뭔소린지 잘모르겠는데 이런게 있다고 함 ㅇㅇ

"""핸드폰 번호 가림 처리하기"""
def coding_phone_number(phone_number):
    phone_number_segment_A = phone_number[:-4]
    phone_number_segment_B = phone_number[-4:]
    phone_number_segment_A = '*'*len(phone_number_segment_A)
    phone_number = phone_number_segment_A + phone_number_segment_B
    return phone_number
# print(coding_phone_number('01033334444'))

"""행렬의 덧셈"""
#1 numpy로 더한 다음 그냥 .tolist() 해도 되지만, 이 문제의 요지는 그게 아니다.

#2 서로 다른 2차원 행렬 안의 요소끼리의 접근시키기--zip을 통해 간단하게 다음과 같이 서로 묶을 수 있다.
def sumMatrix(A,B):
    answer = [[c + d for c, d in zip(a, b)] for a, b in zip(A,B)]
    return answer

"""Summer/Winter Coding 2018 예산 분배하기"""
def desseminating_budget(d, budget):
    count =0
    d.sort()
    for money in d:
        if budget>=0:
            budget -=money
            count+=1
    if budget<0:
        count-=1
    return count

"""2018 KAKAO BLIND RECRUITMENT [1차] 비밀지도"""
#1
"""numpy를 사용한 풀이라서 비주얼 스튜디오 코드에서는 주석처리함"""
# import numpy as np
# def secret_map1(n, arr1, arr2):
#     for i in range(len(arr1)):
#         if arr1[i] == 0:
#             res_str = '0'*n
#             arr1[i] = res_str
#             continue
#         number_to_convert = arr1[i]
#         binary_digits =0
#         while 2**binary_digits <= number_to_convert:
#             binary_digits+=1
        
#         res_str = ''
#         for iterator in range(binary_digits,0,-1):
#             count = number_to_convert //(2**(iterator-1))
#             number_to_convert %= (2**(iterator-1))
            
#             res_str += str(count)
#         while len(res_str) < n:
#             res_str = '0' + res_str
#         arr1[i] = res_str
#     for i in range(len(arr2)):
#         if arr2[i] == 0:
#             res_str = '0'*n
#             arr2[i] = res_str
#             continue
#         number_to_convert = arr2[i]
#         binary_digits =0
#         # number_to_convert
#         while 2**binary_digits <= number_to_convert:
#             binary_digits+=1
#             res_str = ''
#         for iterator in range(binary_digits,0,-1):
#             count = number_to_convert //(2**(iterator-1))
#             number_to_convert %= (2**(iterator-1))
            
#             res_str += str(count)

#         while len(res_str) < n:
#             res_str = '0' + res_str
#         arr2[i] = res_str
    
#     for i in range(n):
#         arr1[i] = list(int(x) for x in arr1[i])
#         arr2[i] = list(int(x) for x in arr2[i])
#     arr1 = np.array(arr1)
#     arr2 = np.array(arr2)
    
#     conducted_array = (arr1+arr2).tolist()
#     for i in range(n):
#         for j in range(n):
#             if conducted_array[i][j] >0:
#                 conducted_array[i][j] = '#'
#             else:
#                 conducted_array[i][j]= ' '
#     for i in range(n):
#         conducted_array[i] = "".join(conducted_array[i])
#     return conducted_array

#2
"""위 문제의 현명한 방법"""
def secret_map2(n, arr1, arr2):
    answer = []
    for i,j in zip(arr1,arr2):
        a12 = str(bin(i|j)[2:]) #킬링포인트1: 0bXXXXXX꼴로 이진법 변환해주는 함수 bin + 어짜피 둘다 0이어야 되므로 합쳐서 이진법 변환해 버리는 형태
        a12=a12.rjust(n,'0') #킬링포인트2: 입력값으로 개수를 맞출 때까지 채워주는 매소드 rjust(길이, 채울 요소)
        a12=a12.replace('1','#') #킬링포인트3: 그대로 문자열로 가공하기 - replace로
        a12=a12.replace('0',' ')
        answer.append(a12)
    return answer
 

"""2019 KAKAO BLIND RECRUITMENT : 실패율"""
class GameStages:
    def __init__(self, all_list, current_level,users_by_far):
        self.all_list = all_list
        self.current_level = current_level
        self.failure = all_list.count(current_level)/users_by_far

def failure_percentage(N, stages):
    GameStages_list = []
    num_of_users_by_far =len(stages)
    for i in range(1, N+1):
        if num_of_users_by_far !=0:
            GameStages_list.append(GameStages(stages, i,num_of_users_by_far))
        else:
            GameStages_list.append(GameStages(stages, i,1))
        num_of_users_by_far -= stages.count(i)
    answer = []
    GameStages_list.sort(key = lambda x: (-x.failure, x.current_level))
    for stage in GameStages_list:
        answer.append(stage.current_level)
    return answer
# print(failure_percentage(4,[4,4,4,4,4]))
# print(failure_percentage(5, [2, 1, 2, 6, 2, 4, 3, 3]))


"""2018 KAKAO BLIND RECRUITMENT [1차] 다트 게임"""
#1 무수한 노가다의 if문으로 무엇이든 할 수 있어!
def dart_game(game_result):
    calculating_list =[]
    phase =-1
    for i in range(len(game_result)):
        if game_result[i] in ['2','3','4','5','6','7','8','9']:
            calculating_list.append(int(game_result[i]))
            phase+=1
        if game_result[i] == '1':
            if game_result[i+1] == '0':
                calculating_list.append(10)
                phase+=1
            else:
                calculating_list.append(1)
                phase+=1
        if game_result[i] == '0':
            try:
                if game_result[i-1] != '1':
                    calculating_list.append(0)
                    phase+=1
            except:
                calculating_list.append(0)
                phase+=1
        
        elif game_result[i] =='S':
            pass
        elif game_result[i] == 'D':
            calculating_list[phase] = calculating_list[phase]**2
        elif game_result[i] == 'T':
            calculating_list[phase] = calculating_list[phase]**3
        
        elif game_result[i] == '*':
            if phase == 0:
                calculating_list[phase] = calculating_list[phase]*2
            else:
                calculating_list[phase-1] = calculating_list[phase-1]*2
                calculating_list[phase] = calculating_list[phase]*2
        elif game_result[i] == '#':
            calculating_list[phase] = calculating_list[phase]*(-1)
    return sum(calculating_list)
# print(dart_game('1S2D*3T'))

"""위 문제를 정규표현식으로 푼 것. 정규표현식을 잘 활용하면 훨씬 편할 듯"""
#2 출제의도 -- 정규표현식과 re모듈
import re
def dart_game_genuine(dartResult):
    bonus = {'S':1, "D":2, "T":3}
    option = {'':1,'*':2,'#':-1} #최종 계산시 dartResult에서 탐색하여 꺼내올 계산용 딕셔너리

    p = re.compile('(\d)([SDT])([*#]?)') #정규표현식 0-9와, SDT와, *#이 있어도되고없어도되는것 의 꼴을 가진 것이 p이다.
    dart = p.findall(dartResult) #p의 형태를 가진 것을 dartResult에서 찾아 list에 반환하는데, 이때 어떤 형태 p에 해당하는 것인지도 리스트에 나타남

    for i in range(len(dart)):
        if dart[i][2] == '*' and i>0:#문제 조건--*은 첫 번째가 아니면 그전조건에도 영향을 미침
            dart[i-1] *=2
        dart[i] = int(dart[i][0]) ** bonus[dart[i][1]] * option[dart[i][2]] #최종 계산식. 공간효율적으로 dart리스트에 재정의
    return sum(dart)
# print(dart_game_genuine('1S2D*3T'))

"""프로그래머스 Lv2, 스택, 큐: 다리를 지나는 트력 2try"""
class Trucks:
    def __init__(self, weight, bridge_length):
        self.weight = weight
        self.status = 0
        self.bridge_length = bridge_length
        self.passed = False
    def move(self):
        if self.status <= self.bridge_length:
            self.status+=1
        if self.status >= self.bridge_length:
            self.passed = True
    
def crossing_bridge(bridge_length, weight, truck_weights):
    time =0
    bridge_list =[]
    finished_list = []
    current_weight =0
    Trucks_list = []
    truck_weights.sort()
    full_number = len(truck_weights)

    #트럭 생성
    for i in range(len(truck_weights)):
        Trucks_list.append(Trucks(truck_weights[i], bridge_length))
    
    while len(finished_list) < full_number:
        try:
            if current_weight + Trucks_list[0].weight <= weight:
                current_weight += Trucks_list[0].weight
                bridge_list.append(Trucks_list.pop(0))
                for truck in bridge_list:
                    truck.move()
                    try:
                        if truck.passed:
                            finished_list.append(bridge_list.pop(0))
                            current_weight -= truck.weight
                    except:
                        pass
                time+=1
            else:
                for truck in bridge_list:
                    truck.move()
                    try:
                        if truck.passed:
                            finished_list.append(bridge_list.pop(0))
                            current_weight -= truck.weight
                    except:
                        pass
                time+=1
        except:
            for truck in bridge_list:
                    truck.move()
                    try:
                        if truck.passed:
                            finished_list.append(bridge_list.pop(0))
                            current_weight -= truck.weight
                    except:
                        pass
            time+=1
    if full_number ==1:
        time+=1
    return time
# print(crossing_bridge(2,10,[7,4,5,6]))

"""다리를 지나는 트럭 : 문제 해설"""
def solution(bridge_length, weight, truck_weights):
    time = 0 #시간 변수
    q = [0] * bridge_length #다리 구현
    
    while q: #다리가 존재하는 동안
        time += 1 #1초간 이하의 일들이 일어난다.
        q.pop(0) #한칸씩 전진
        if truck_weights: #조건 1: 트럭이 아직 남아 있다면
            if sum(q) + truck_weights[0] <= weight: #조건2-1 해당 트럭의 무게가 올라가도 한계를 초과하지 않는다면  ++ 무게를 sum으로 구현
                q.append(truck_weights.pop(0)) #트럭에서 하나 빼와서 다리에 싣는다.
            else:
                q.append(0) #조건 2-2 해당 트럭이 올라갔을 때 무게가 한계를 초과한다면 한칸 씩 전진하는 것을 그냥 0을 넣음으로서 구현
    
    return time

"""백준으로 잠시(?) 건너왔습니다"""

"""2577 숫자의 개수"""
# num_list = []
# for i in range(3):
#     num_list.append(int(input()))

# D=1
# for num in num_list:
#     D *=num

# D = str(D)
# for i in range(10):
#     print(f"{str(D).count(str(i))}")

"""3052 나머지"""
#입력 부분
# num_list =[]
# for i in range(10):
#     num_list.append(int(input()))

def leftovers(num_list):
    leftover_list =[]
    for i in range(len(num_list)):
        leftover_list.append(num_list[i]%42)
    leftover_list = set(leftover_list)
    return len(leftover_list)

# print(leftovers(num_list))

"""1546 평균"""
# N = int(input())
# score_list =input().split()
# for i in range(len(score_list)):
#     score_list[i] = int(score_list[i])

def manipulating_average(score_list):
    max_score = max(score_list)
    for i in range(len(score_list)):
        score_list[i] = score_list[i]/max_score*100
    return sum(score_list)/len(score_list)
# print(manipulating_average(score_list))    

"""4344 평균은 넘겠지"""

# N = int(input())
# entire_list=[]
# for i in range(N):
#     entire_list.append(list(input().split()))

# # def it_would_be_better_than_avg(entire_list):
# ratio_list=[]

# for i in range(len(entire_list)):
#     entire_list[i] = entire_list[i][1:]

# for i in range(len(entire_list)):
#     for j in range(len(entire_list[i])):
#         entire_list[i][j] = int(entire_list[i][j])

# for i in range(len(entire_list)):
#     count=0
#     for j in range(len(entire_list[i])):
#         if entire_list[i][j] > sum(entire_list[i])/len(entire_list[i]):
#             count+=1
#     ratio_list.append(str(round(count/len(entire_list[i])*100,3)))

# for i in range(len(ratio_list)):
#     while len(ratio_list[i])<6:
#         ratio_list[i] = ratio_list[i]+'0'
#     print(f"{ratio_list[i]}%")

"""셀프 넘버 4673"""
"""boolean으로 이루어진 리스트를 좀 더 활용하는 마인드"""
# total_num [True]*10000
# for i in range(1,len(total_num)):
#     str_i = str(i)
#     temp=0
#     for j in range(len(str(i))):
#         temp+=int(str_i[j])
#     new_num = i+temp
#     if new_num<10000 and total_num[new_num]:
#         total_num[new_num] = False
# for i in range(len(total_num)):
#     if total_num[i] and i!=0:
#         print(i)

"""1065 한수"""
# N = int(input())
def han_number(N):
    if N>=10:
        count=9 #이미 한 자리 수는 다 포함임
        for x in range(10, N+1):
            boo = True
            str_x = str(x)
            d = int(str_x[1]) - int(str_x[0])
            for i in range(len(str_x)-1):
                if (int(str_x[i+1])-int(str_x[i])) != d:
                    boo = False
                    break
            if boo:
                count+=1
    else:
        count=N
    return count
# print(han_number(N))

"""11654 아스키 코드"""
def turn_into_ascii(X):
    return ord(X)

# X= input()
# print(turn_into_ascii(X))

"""1157 단어 공부"""
import sys
def word_studying(s):
    s=s.upper()
    keys = set(s)
    max_letter = s[0]
    max_count = s.count(max_letter)
    for key in keys:
        if s.count(key) > max_count:
            max_count = s.count(key)
            max_letter = key
    for key in keys:
        if s.count(key) == max_count and key != max_letter:
            return '?'
    else:
        return max_letter
# s = sys.stdin.readline()
# print(word_studying(s))

"""1152 단어 개수"""
import sys
# import re
# s = sys.stdin.readline()
def word_counting(s):
    if s.isspace():  #null문자열이 아니라, 공백 \t \n 등- 으로 이루어져 있는지 확인하는 메소드 .isspace()
        return 0
    else:
        s=s.strip()
        my_list = s.split(' ')
        return len(my_list)

# print(word_counting(s))

"""2908 상수"""
def constant(A,B):
    str_A = str(A)
    str_B = str(B)
    str_A = str_A[::-1]
    str_B = str_B[::-1]

    return max(int(str_A), int(str_B))

# A, B = map(int, input().split())
# print(constant(A,B))

"""다이얼 5622"""
#정규 표현식으로 풀려 하니 더 어렵다!
# s = input()
# import re
# def dialect(s):
#     grandma_memory = {'A':2, 'B':2,'C':2, 'D':3, 'E':3, 'F':3, 'G':4, 'H':4, 'I':4, 'J':5, 'K':5, 'L':5, "M":6, 'N':6, 'O':6, "P":7, "Q":7, "R":7,
#     "S": 7, "T":8, "U":8, "V":8, 'W':9, 'X':9, 'Y':9, 'Z':9}

#     q = re.compile('([ABC]?)([DEF]?)([GHI]?)([JKL]?)([MNO]?)([PQRS]?)([TUV]?)([WXYZ]?)')
#     number = q.findall(s)
#     # for i in range(len(number)):
#     #     number[i] = grandma_memory[number[i][]]
#     return number
# print(dialect('UNUCIC'))

# print(ord('U'))
# s = input()
def dialect2(s):
    grandma_memory = {'A':2, 'B':2,'C':2, 'D':3, 'E':3, 'F':3, 'G':4, 'H':4, 'I':4, 'J':5, 'K':5, 'L':5, "M":6, 'N':6, 'O':6, "P":7, "Q":7, "R":7,
    "S": 7, "T":8, "U":8, "V":8, 'W':9, 'X':9, 'Y':9, 'Z':9}
    time =0
    for i in range(len(s)):
        time += (grandma_memory[s[i]]+1)
    return time
# print(dialect2(s))

"""2941 크로아티아 알파벳"""
def croatia_alphabet(s):
    croatia_dict = {'c=': 1, 'c-':1, 'dz=': 1, 'd-':1, 'lj':1, 'nj':1, 's=':1, 'z=':1}
    count =0
    left_over =0
    for i in range(len(s)):
        for key in croatia_dict.keys():
            if key in s[:i]:
                count+=1
                q = s[:i]
                s = s[i:]
                q= q.replace(key, '')
                left_over += len(q)
                break
    count+=left_over
    count+=len(s)
    return count

# s = input()
# print(croatia_alphabet(s))

croatia_dict = {'c=': 1, 'c-':1, 'dz=': 1, 'd-':1, 'lj':1, 'nj':1, 's=':1, 'z=':1}
# s = input()
# for i in croatia_dict.keys():
    # s = s.replace(i, '!') #--중간에 이렇게 아예 다른 문자열 놓으면 1: 없어짐으로 인해 발생하는 추가 단어 생성 방지, 2:길이로 한꺼번에 답 도출 가능
# print(len(s))

"""1316 그룹 단어 체커"""
class GroupWord:
    def __init__(self, letter):
        self.letter = letter
        self.written = False
        self.group_word = True

# N = int(input())
# word_list = []
# for i in range(N):
#     word_list.append(input())

def groupword_checker(word_list):
    count =len(word_list)
    for word in word_list:
        if len(word) == 1:
            continue
        class_dict ={}
        set_of_letters = set(word)
        for letter in set_of_letters:
            class_dict[letter]=GroupWord(letter)


        for i in range(len(word)-1):
            if word[i] != word[i+1]:
                if class_dict[word[i]].written == False:
                    class_dict[word[i]].written = True

                if class_dict[word[i+1]].written == True:
                    class_dict[word[i+1]].group_word = False
                    break
        if class_dict[word[-1]].written == True:
            if class_dict[word[-1]] != class_dict[word[-1]]:
                class_dict[word[-1]].group_word = False
        if word.index(word[-1]) == (len(word)-1):
            class_dict[word[-1]].group_word = True
        
        for instance in class_dict.values():
            if instance.group_word == False:
                count -=1
                break
    return count
# print(groupword_checker(word_list))

"""1712 sonik boongi"""

# A,B,C = map(int, input().split())

def sonsik_boongi_point(A,B,C):
    if B>=C:
        return -1
    return int(A/(C-B))+1
# print(sonsik_boongi_point(A,B,C))

"""2447 별 찍기"""
# N = int(input())
# def hole_square(N):
#     if N == 3:
#         return "***\n* *\n***"
#     else:
#         answer = hole_square(N/3)+=(hole_square(N/3)*2)
#         return answer
# print(hole_square(N))

"""2108 평균"""
# #import numpy as np
# N = int(input())
# num_list =[]
# for i in range(N):
#     num_list.append(int(input()))
# # num_list = [-1,-1,-2,-2,-3]
# # num_list = [1,3,8,-2,2]
# # num_list = [4000]

# answer = []


# #1산술 평균
# answer.append(int(round(np.mean(num_list),0)))

# #2중앙값
# answer.append(int(np.median(num_list)))

# #최빈값
# class Number_in_list:
#     def __init__(self, number):
#         self.number = number
#         self.count =0

# Number_list = []
# num_set = set(num_list)
# for num in num_set:
#     Number_list.append(Number_in_list(num))
# for instances in Number_list:
#     instances.count += num_list.count(instances.number)
# Number_list.sort(key = lambda x : (-x.count, x.number))
# key = Number_list[0].count
# Boo=True
# for i in range(1,len(Number_list)):
#     if Number_list[i].count == key:
#         answer.append(Number_list[1].number)
#         Boo = False
#         break
# if Boo:
#     answer.append(Number_list[0].number)

# #4범위
# import heapq as h
# h.heapify(num_list)
# answer.append(max(num_list)-h.heappop(num_list))

# for avg in answer:
#     print(avg)

"""1427 소트인사이드"""
# S = input()
def sort_inside(S):
    S=list(S)
    S.sort(reverse = True)
    return "".join(S)
# print(sort_inside(S))

"""11650 좌표 정렬하기"""
# import sys
# # N = int(sys.stdin.readline())
# coord_list = []
# for i in range(N):
#     A, B = map(int, sys.stdin.readline().split())
#     coord_list.append((A,B))

# coord_list.sort(key = lambda x : (x[0],x[1]))
 
# for coord in coord_list:
#     print(f"{coord[0]} "+ f"{coord[1]}")


"""11651 좌표 정렬하기 2"""
# import sys
# N = int(sys.stdin.readline())
# coord_list = []
# for i in range(N):
#     A,B = map(int, input().split())
#     coord_list.append((A,B))
# coord_list.sort(key = lambda x : (x[1], x[0]))
# for coord in coord_list:
#     print(f"{coord[0]} {coord[1]}")


"""1181 단어 정렬"""
# import sys
# N = int(sys.stdin.readline())
# string_list =[]
# for i in range(N):
#     string_list.append(sys.stdin.readline())
# string_list = set(string_list)
# string_list = list(string_list)
# string_list.sort(key = lambda x : (len(x), x))
# res_str=""
# for word in string_list:
#     res_str += word
    
# print(res_str)

"""10814 나이순 정렬"""
# import sys
# N = int(sys.stdin.readline())
# member_list =[]
# for i in range(N):
#     A,B = sys.stdin.readline().split()
#     member_list.append((A,B,i))
# member_list.sort(key = lambda x : (int(x[0]), x[2]))
# for human in member_list:
#     print(f"{human[0]} {human[1]}")


"""10989 수 정렬하기 3"""
#파이썬 sort메소드는 시간복잡도가 N lgN 이다. N이나 lgN급으로 낮추려면??
#hint: 카운팅 소트
"""
수의 대소를 비교하는 comparison 기반의 정렬은 아무리 잘 해도 N lgN이 최대로 낮아진 형태이다.
그러나 non-comparison 기반의 counting sort는 N 까지 낮출 수 있다.
"""
# Counting Sort Algorithmn ---숫자만 쓸 수 있음
target_array = [2,0,1,4,5,4,3,2,0,1,1,0,5,4,3]
counting_array = [3,3,2,2,3,2] #각 수=인덱스 | 리스트에 저장된 값: 각 수의 개수
counting_array = [3,6,8,10,13,15] #counting_array를 누적해서 다시 업데이트(직전 요솟값을 더하여)
answer = [-1] * len(target_array) #정렬된 배열을 담을 리스트 생성
#현재 누적 counting_array의 각 값이 의미하는 바는: i번째 인덱스와 그 값에 대하여:
# 정렬시: target array에 숫자 i가 인덱스 range(counting_array[i-1],counting_array[i]) 차지하고 있다는 뜻
#이 의미를 이해한 상태에서, 역순으로 target_array의 값을 answer에 채워넣으면 된다.
#단, 숫자 한개가 들어갈 때마다 counting_array의 해당 수에 해당하는 값을 -1씩 해주면 자연스럽게 채워진다.

def counting_sort(array):
    counting_array =[0]*(max(array)+1) #max는 일반 탐색 메소드로 시간복잡도 N이다.
    for i in range(len(counting_array)):
        try:
            counting_array[i] = array.count(i)
        except:
            pass
    for i in range(1,len(counting_array)):
        counting_array[i] = counting_array[i-1]+counting_array[i]
    
    answer = [-1]*len(array)
    for i in range(len(array)-1, -1, -1):
        answer[counting_array[array[i]]-1] = array[i]
        counting_array[array[i]]-=1

    return answer
# print(counting_sort([2,0,1,4,5,4,3,2,0,1,1,0,5,4,3]))
"""대박!"""  

"""counting_sort for 10989""" #반환하는 값이 구조적으로 같다면, 딱히 소팅할 필요가 없다.
# import sys
# N = int(sys.stdin.readline())
# dic = {}
# for i in range(N):
#     A = int(sys.stdin.readline())

#     if A in dic:
#         dic[A] +=1
#     else:
#         dic[A] = 1
# for num in sorted(dic.items()): #items() /keys() / values() --리스트와 같지만. dict_**** 자료형이므로 리스트 메소드를 사용X -- 사용하고 싶다면 list(a.items()) 이러면 된다
#     for i in range(num[1]):#num[1]은 각 요소가 몇 개인지 나타내 주는 딕셔너리값임!
#         print(num[0])#그 개수만큼 해당 수를 출력



        