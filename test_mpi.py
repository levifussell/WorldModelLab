from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

def run(conn: Connection):
    conn.send("hello.")
    conn.send("goodbye.")

if __name__ == "__main__":
    p, c = Pipe()
    pro = Process(target=run, args=(c,))
    pro.start()
    pro.join()
    d = p.recv()
    print(d)
    d = p.recv()
    print(d)

    # too many 'sends' is blocking. Pipe is max 512 bytes.
    for i in range(1000):
        print(i)
        p.send("H")
