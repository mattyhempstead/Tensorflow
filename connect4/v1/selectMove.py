import random

def maxMove(moveRanks):
    '''
        Returns the highest ranked move
    '''
    return moveRanks.argmax()


def odds(moveRanks):
    '''
        Returns a move with odds equal to moveRanks array
    '''
    rand = sum(moveRanks) * random.random()
    for i,k in enumerate(moveRanks):
        rand -= k
        if rand <= 0:
            return i


def maxMoveProb(moveRanks, p=1):
    '''
        Returns the max move with probability p
        If this fails, return next largest move with probability p
        Repeat until last move and then return last move
    '''
    moveRanksCopy = moveRanks.copy()
    while True:
        m = moveRanksCopy.argmax()
        if random.random() <= p:
            return m
        
        moveRanksCopy[m] = 0

        # If all moves have now been set to zero, return last selected move
        if max(moveRanksCopy) == 0:
            return m