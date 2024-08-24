def sort(arr):
    gap = len(arr)-1
    while gap > 0:
        for i in range(len(arr)):            
            if i + gap == len(arr):
                break
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
        gap = gap - 1
    return arr