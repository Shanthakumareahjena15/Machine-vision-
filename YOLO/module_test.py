# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:04:35 2020

@author: shanthakumar
"""



def fish_count(outputs):
    dummy_scores = []
    dummy_nums = []
    boxes, scores, classes, nums = outputs
    boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]
    for i in range(nums):
        if (scores[i]*100) > 75:
            dummy_scores.append(scores)
            dummy_nums.append(nums)
    
    
    print(dummy_scores)
    print('######')
    print(dummy_nums)
    
