from __future__ import division
from builtins import range, zip
from future.utils import viewitems

from collections import deque
import random
import functools
import itertools
import warnings
from collections import defaultdict



def test():
    import unittest
    import dedupe
    import dedupe.sampling
    import dedupe.predicates
    import dedupe.api
    from collections import deque
    import random
    
    data_dict = {    '1' : {'name' : 'Bob',         'age' : '51'},
                     '2' : {'name' : 'Linda',       'age' : '50'},
                     '3' : {'name' : 'Gene',        'age' : '12'},
                     '4' : {'name' : 'Tina',        'age' : '15'},
                     '5' : {'name' : 'Bob B.',      'age' : '51'},
                     '6' : {'name' : 'bob belcher', 'age' : '51'},
                     '7' : {'name' : 'linda ',      'age' : '50'} }
    
    
    predicates = [dedupe.predicates.SimplePredicate(dedupe.predicates.sameThreeCharStartPredicate,
                                             'name'),
                  dedupe.predicates.SimplePredicate(dedupe.predicates.nearIntegersPredicate, 'age')]
    sampler = dedupe.sampling.dedupeSamplePredicates
    return dedupe.sampling.blockedSample(sampler, 6, predicates, deque(data_dict.items()))


def blockedSample(sampler, sample_size, predicates, *args) :
    '''
    sampler: sampler object to use
    sample_size: max number of samples to generate (will split evenly by predicate)
    predicates: list of predicates
    '''

    blocked_sample = set()
    remaining_sample = sample_size - len(blocked_sample) # Number of samples left to generate
    previous_sample_size = 0

    # Usually only one iteration
    while remaining_sample and predicates:
        
        # Randomize predicate order
        random.shuffle(predicates)

        new_sample = sampler(remaining_sample, 
                             predicates,
                             *args)

#        # Keep only non empty samples
#        filtered_sample = (subsample for subsample 
#                           in new_sample if subsample)
#        blocked_sample.update(itertools.chain.from_iterable())

#        predicates = [pred for pred, pred_sample 
#                      in zip(predicates, new_sample)
#                      if pred_sample or pred_sample is None]        
        
        
        new_predicates = []
        for pred, subsample in zip(predicates, new_sample):
            if subsample:
                blocked_sample.update(subsample)
            if subsample or subsample is None:
                new_predicates.append(pred)
        predicates = new_predicates            

        growth = len(blocked_sample) - previous_sample_size # How many samples we added
        growth_rate = growth/remaining_sample # Ratio of objective reached

        remaining_sample = sample_size - len(blocked_sample)
        previous_sample_size = len(blocked_sample)


        
        # If it will be to long 
        if growth_rate < 0.001 :
            warnings.warn("%s blocked samples were requested, "
                          "but only able to sample %s"
                          % (sample_size, len(blocked_sample)))
            break
    return blocked_sample

def dedupeSamplePredicates(sample_size, predicates, items) :
    n_items = len(items)

    for subsample_size, predicate in subsample(sample_size, predicates) : 

        if not subsample_size :
            yield None
            continue

        items.rotate(random.randrange(n_items))
        items.reverse()

        yield dedupeSamplePredicate(subsample_size,
                                    predicate,
                                    items)

def dedupeSamplePredicate(subsample_size, predicate, items) :

    sample = []
    block_dict = {}
    predicate_function = predicate.func
    field = predicate.field

    for pivot, (index, record) in enumerate(items) :
        column = record[field]
        if not column :
            continue

        if pivot == 10000:
            if len(block_dict) + len(sample) < 10 :
                return sample

        
        block_keys = predicate_function(column)
        
        for block_key in block_keys:
            if block_key not in block_dict :
                block_dict[block_key] = index
            else :
                pair = sort_pair(block_dict.pop(block_key), index)
                sample.append(pair)
                subsample_size -= 1

                if subsample_size :
                    break
                else :
                    return sample

    else :
        return sample

def linkSamplePredicates(sample_size, predicates, items1, items2) :
    n_1 = len(items1)
    n_2 = len(items2)
    
    # Subsample_size is basically the number of samples for each predicate
    for subsample_size, predicate in subsample(sample_size, predicates) :
        if not subsample_size :
            yield None
            continue

        try:
            items1.rotate(random.randrange(n_1))
            items2.rotate(random.randrange(n_2))
        except ValueError :
            raise ValueError("Empty itemset.")

        try :
            items1.reverse()
            items2.reverse()
        except AttributeError :
            items1 = deque(reversed(items1))
            items2 = deque(reversed(items2))

        # yield linkSamplePredicate(subsample_size, predicate, items1, items2)

        temp = linkSamplePredicate(subsample_size, predicate, items1, items2)
        print('predicate ', predicate, 'has ', len(temp), ' samples')
        yield temp


def linkSamplePredicate(subsample_size, predicate, items1, items2):
    '''Return a list of max sumbsample_size samples blocked using the predicate'''
    sample = []

    predicate_function = predicate.func
    field = predicate.field

    # dicts where key is a block index and value is list of indexes
    red = defaultdict(list) # For items1
    blue = defaultdict(list) # For items2

    for i, (index, record) in enumerate(interleave(items1, items2)):
        if i == 20000:
            if min(len(red), len(blue)) + len(sample) < 10 :
                return sample
        
        column = record[field] # column is a value for the field
        if not column:
            red, blue = blue, red
            continue

        block_keys = predicate_function(column) # 
        for block_key in block_keys: # For the given record, see if any of the block indexes are pre-existing
            if blue.get(block_key): # if block key also exists in other dataset
            
                # TODO: pop(0) so it is less likely to create the same sample as previously
                pair = sort_pair(blue[block_key].pop(), index) # Take last index found to make pair
                sample.append(pair)

                subsample_size -= 1
                if subsample_size :
                    break # Continue iterating through records
                else :
                    return sample # We have all the samples we neef for this predicate
            else:
                red[block_key].append(index) # 1st round: add index of record with its blocking key

        red, blue = blue, red

    print('WE GOT HERE to look for more samples')
    # Continue looking in itmes2
    for index, record in itertools.islice(items2, len(items1)): # same as items2[:len(items1)]
        column = record[field]
        if not column :
            continue

        block_keys = predicate_function(column)
        for block_key in block_keys:
            if red.get(block_key):
                pair = sort_pair(red[block_key].pop(), index)
                sample.append(pair)

                subsample_size -= 1
                if subsample_size :
                    break
                else :
                    return sample

    return sample

def evenSplits(total_size, num_splits):
    '''
    Yields num_splits integers such that the values are close to 
    total_size/num_splits cummulative sum tends towards average total_size
    '''
    avg = total_size/num_splits
    split = 0
    for _ in range(num_splits) :
        split += avg - int(split)
        yield int(split)

def subsample(total_size, predicates) :
    splits = evenSplits(total_size, len(predicates))
    for split, predicate in zip(splits, predicates) :
        yield split, predicate

def interleave(*iterables) :
    '''
    Returns iterable with values that alternate btw iterables
    Ex: interleave(range(5), range(10,15)))
    -> [0, 10, 1, 11, 2, 12, 3, 13, 4, 14]
    '''    
    return itertools.chain.from_iterable(zip(*iterables))

def sort_pair(a, b):
    if a > b :
        return (b, a)
    else :
        return (a, b)

def randomDeque(data) :
    data_q = deque(random.sample(viewitems(data), len(data)))
    
    return data_q


dedupeBlockedSample = functools.partial(blockedSample, dedupeSamplePredicates) 
linkBlockedSample = functools.partial(blockedSample, linkSamplePredicates) 


