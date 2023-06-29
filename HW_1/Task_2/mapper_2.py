#! /user/bin/python3

import sys

chunk_size = 0
numbers = []
data = sys.stdin
price_sum = 0

for line in data:
    prices_str = line.strip().split(",")
    if len(prices_str) > 8:
        price = float(prices_str[9])
        numbers.append(price)
        price_sum += price
        chunk_size += 1

mk = price_sum / len(numbers) #mean
vk = 0 #variance


for value in numbers:
    vk += (value - mk) ** 2

if numbers:
    dispersion = vk / (len(numbers) - 1)
    result = (chunk_size, mk, vk)
    print(result)
