import sys

data = sys.stdin.readlines()
ck = 0
mk = 0
vk = 0
count = 0

for n, m, v in data:
    if count == 0:
        ck, mk = n, m
        vk = v
    else:
        mk = (ck*mk + n*m)/(ck + n)
        ck += n
        vk = (vk*mk + n*v)/(ck + n) + ((vk*v) - (m*mk)/(ck + n))
    count += 1

y = count, mk, vk
print(y)
