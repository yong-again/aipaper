from seq2seq import Seq2seq


def main():
    seq2seq = Seq2seq(lr=0.3, init_range=0.3)
    
    for i in range(1000):
        cost = seq2seq.train([2, 1], [2])
        cost += seq2seq.train([1], [1])
        cost += seq2seq.train([3, 1], [3])
        
        if i % 100 == 0:
            print('Epoch:', i)
            print('training cost: ', cost / 3)
            
            print([2, 1], '->', seq2seq.predict([2, 1]))
            print([1], '->', seq2seq.predict([1]))
            print([3, 1], '->', seq2seq.predict([3, 1]))
            
if __name__ == "__name__":
    main()