'''
name       : nlp2xml.py
author     : [redacted]
authored   : 5 May 2020
updated    : 9 June 2020
description: translates annotated txt -> xml Python object
usage      : python nlp2promela.py rfcs-annotated-tidied/TCP.xml
'''
import sys
import json
import os

from stringUtils      import cleanFile
from fsmUtils         import FSM
from xmlUtils         import *
from printUtils       import printTransition, printHeuristicRemoval
from testConstants    import *
# from testPromelaModel import *

# INPUT:
#     we expect only a single argument, namely the path to the rfc .txt file
# OUTPUT:
#     a .png file in the directory from which the script was run, with the same
#     name as the input file, modulo extension.  This image represents
#     graphically the FSM of the protocol, or at least, my best guess.
def main(writepromela=False, writepng=True):

    name = sys.argv[1]
    file = "./rfcs-definitions/def_states.txt"
    states  = []
    with open(file, 'r') as fp:
        for line in fp:
            if name in line:
                list = line.split("	")
                states.append(list[1])
    if name == "OSPF":
        states.append("Unknown")
    initial = states[0]
    transitions = []

    event2id = []
    if name == "BGPv4":
        event2id = {'ManualStart': '1', 'ManualStop': '2', 'AutomaticStart': '3', 'ManualStart_with_PassiveTcpEstablishment': '4', 'AutomaticStart_with_PassiveTcpEstablishment': '5', 'AutomaticStart_with_DampPeerOscillations': '6', 'AutomaticStart_with_DampPeerOscillations_and_PassiveTcpEstablishment': '7', 'AutomaticStop': '8', 'ConnectRetryTimer_Expires': '9', 'HoldTimer_Expires': '10', 'KeepaliveTimer_Expires': '11', 'DelayOpenTimer_Expires': '12', 'IdleHoldTimer_Expires': '13', 'TcpConnection_Valid': '14', 'Tcp_CR_Invalid': '15', 'Tcp_CR_Acked': '16', 'TcpConnectionConfirmed': '17', 'TcpConnectionFails': '18', 'BGPOpen': '19', 'BGPOpen with DelayOpenTimer running': '20', 'BGPHeaderErr': '21', 'BGPOpenMsgErr': '22', 'OpenCollisionDump': '23', 'NotifMsgVerErr': '24', 'NotifMsg': '25', 'KeepAliveMsg': '26', 'UpdateMsg': '27', 'UpdateMsgErr': '28'}


    trans_path = "./trans_data/{}.json".format(name.lower())
    with open(trans_path, 'r') as fp:
        blocks = json.load(fp)
        for block in blocks:
            for trans in block["transition"]:
                cur_state = trans["cur_state"]
                new_state = trans["new_state"]
                events = trans["event"]
                event_list = events.split('\n')
                for event in event_list:
                    if name == "BGPv4":
                        transitions.append((cur_state, [event2id[event]], new_state))
                    else:
                        transitions.append((cur_state, [event], new_state))

    msgs = None

    result = FSM(states,
                 initial,
                 transitions,
                 labeled=False,
                 msgs=msgs)

    result.save(writepng, writepromela, name)

if __name__ == "__main__":
    main()
