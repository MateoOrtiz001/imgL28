def areDigitsDifferent(arr):
    gap = len(arr)-1
    while gap > 0:
        for i in range(len(arr)): 
            if i + gap == len(arr):
                break           
            if arr[i] == arr[i + gap]:
                return False
        gap = gap - 1
    return True

def areArraysEqual(arrA,arrB):
    if(len(arrA) != len(arrB)):
        return False
    for i in range(len(arrA)):
        if(arrA[i] != arrB[i]):
            return False
    return True

def isTheTranspose(arrA,arrB):
    if(len(arrA) != len(arrB[0]) or len(arrA[0]) != len(arrB)):
        return False
    for i in range(len(arrA)):
        for j in range(len(arrA[0])):
            if(arrA[i][j]!= arrB[j][i]):
                return False
    return True