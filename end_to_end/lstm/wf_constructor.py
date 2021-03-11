import torch
import torch.nn as nn
import random
import itertools
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, :, :])
        return out


class workflow_constructor:
    def __init__(self,model,input_size=1):

        self.softmax = nn.Softmax(dim=0)
        self.model = model
        self.input_size = input_size

    def permutation(self,event_list):
        return [list(event) for event in (itertools.permutations(event_list))]

    def generate_seq(self,start, window_size=10, num_candidates=5, scope=None):
        if isinstance(start, list):
            start = torch.FloatTensor(start).reshape(1, -1)
        bg = start.size(1)
        if scope == None:
            scope = num_candidates
        for i in range(bg, bg + window_size):
            seq = start.clone().detach().view(-1, i, input_size).to(device)
            output = model(seq).cpu()[:, -1, :]
            output = output.reshape(-1)
            predicted = torch.argsort(output)[-num_candidates:]
            nxt = random.randint(1, scope)
            start = torch.cat([start, predicted[-nxt].reshape(1, -1).float()], 1)
        return start, predicted, output

    def predict_next_event(self,seq, num_candidates, ts):
        _, predicted, output = self.generate_seq(seq, 1, num_candidates)
        prob = self.softmax(output)
        scope = 0
        for i in range(num_candidates):
            if prob[predicted[num_candidates - i - 1]] < ts:
                scope = num_candidates - i - 1
                break
        return predicted[scope + 1:].cpu().numpy().tolist()

    def next_possible_event(self,seq, num_candidates=10, ts=0.005, out_of_order=False, bg=0, ed=-1):
        res = []
        if not out_of_order:
            res.extend(self.predict_next_event(seq, num_candidates, ts))
        else:
            seq_list = self.permutation(seq[bg:ed])
            for s in seq_list:
                res.extend(self.predict_next_event(s + seq[ed:], num_candidates, ts))
            res = list(set(res))
        return res

    def is_connected(self,e1, e2, eventmap):
        eventList = list(eventmap.keys())
        visited = {i: False for i in eventList}
        que = []
        que.append(e1)
        while len(que) != 0:
            cur = que.pop(0)
            visited[cur] = True
            if e2 in eventmap[cur]:
                return True
            for e in eventmap[cur]:
                if e in eventList and not visited[e]:
                    que.append(e)
        return False

    def recognize_branch(self,next_event):
        concurrent_group = []
        if 30 in next_event:
            next_event.remove(30)
        event_to_group = [-1 for _ in next_event]
        event_to_next = {event: self.next_possible_event([event]) for event in next_event}
        for i, event in enumerate(next_event):
            if event_to_group[i] == -1:
                cur_group = [event]
                concurrent_group.append(cur_group)
                event_to_group[i] = len(concurrent_group) - 1
            for j, event2 in enumerate(next_event[i + 1:]):
                if self.is_connected(event, event2, event_to_next) and self.is_connected(event2, event, event_to_next):
                    if event_to_group[j + i + 1] == -1:
                        event_to_group[j + i + 1] = event_to_group[i]
                        concurrent_group[event_to_group[i]].append(event2)
        #                 print(concurrent_group,i,j+i+1)
        return concurrent_group

    def recognize_loop(self,seq, concurrent_event):

        next_event = self.next_possible_event(seq)
        for event in next_event:
            if event in seq and event not in concurrent_event:
                if self.find_loop(seq[seq.index(event):]):
                    return seq[seq.index(event):]
        return []

    def find_loop(self,seq):

        loop_len = len(seq)
        loop_seq = [event for event in seq]
        for i in range(loop_len):
            if loop_seq[i] not in self.next_possible_event(loop_seq):
                return False
            loop_seq.append(loop_seq[i])
        return True


    def workflow_construction(self,seq=[0], end={30}, concurrent=False):
        next_event = self.next_possible_event(seq, out_of_order=concurrent, ed=len(seq))
        for event in seq:
            if event in next_event:
                next_event.remove(event)
        print("当前序列为:", seq, "下一个可能出现event为:", next_event)
        if not concurrent:
            loop = self.recognize_loop(seq, [])
            if len(loop) != 0:
                print("当前序列中存在循环结构", loop)
        if set(next_event) == end or set(next_event).issubset(end):
            print("到达终止位置,结束", end)
            return seq
        print("判断分支情况", next_event)
        if len(next_event) >= 2:
            concorrent_group = self.recognize_branch(next_event)

        else:
            concorrent_group = [[next_event[0]]]
        print("发现不同的分支", concorrent_group)
        for group in concorrent_group:
            if len(group) >= 2:
                print("并发执行的分支", group)
                for i, event in enumerate(group):
                    loop = self.recognize_loop([event], group[:i] + group[i + 1:])
                    if len(loop) != 0:
                        print("发现循环", loop)
                self.workflow_construction(group, end, True)
            else:
                self.workflow_construction(group, {30})
        return seq

if __name__=='__main__':
    num_classes = 32
    num_epochs = 300
    batch_size = 2048
    input_size = 1
    model_dir = 'model'
    window_size = 10
    log = 'add_padding_batch_size={}_epoch={}_window_size={}'.format(str(batch_size), str(num_epochs), str(window_size))
    num_layers = 2
    hidden_size = 64
    file_dir = 'data/'
    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_dir + '/' + log + '.pt'))
    model.to(device)
    model.eval()

    wf_cons = workflow_constructor(model,1)
    wf_cons.workflow_construction([0],{30})