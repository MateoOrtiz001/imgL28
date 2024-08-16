def logdosFirstAproximation(n , c= 1,sg = True):
  if sg and n==1 or n==2:     
    return n-1
  n = n>>1                   
  c = c + 1                   
  if n == 2 or n==1:          
    return c
  else:
    return logdosFirstAproximation(n,c,False)  

def logdos(n):
  ln = float.fromhex('0x1.62e42fefa39efp-1')  
  x = logdosFirstAproximation(n)
  if n>2:                                   
    for _ in range(10):  
      x = x - (2**x - n)/(2**x * ln)
  return x

def ln(n):
  ln = float.fromhex('0x1.62e42fefa39efp-1')
  x = logdos(n) * ln
  return x  

def log(n,base = 10):
  if base == 2:
    return logdos(n)
  return ln(n)/ln(base)