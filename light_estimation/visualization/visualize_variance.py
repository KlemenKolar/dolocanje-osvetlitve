import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

preds_heatmap = np.array([[23.0, 6], [17.0, 6], [14.0, 11], [11.0, 9], [9.0, 15], [6.0, 15], [0.0, 15], [0.0, 15], [6.0, 11], [7.0, 15], [10.0, 8], [17.0, 10], [23.0, 11], [26.0, 15], [31.0, 15], [21.0, 8], [17.0, 10], [8.0, 13], [9.0, 15], [23.0, 15], [6.0, 15], [2.0, 15], [25.0, 15], [22.0, 10], [26.0, 15], [27.0, 6], [23.0, 5], [20.0, 6], [14.0, 9], [11.0, 7], [8.0, 6], [10.0, 7], [20.0, 7], [21.0, 10], [20.0, 6], [4.0, 8], [20.0, 6], [13.0, 8], [10.0, 6], [6.0, 10], [11.0, 9], [13.0, 8], [23.0, 9], [30.0, 7], [10.0, 6], [6.0, 6], [4.0, 9], [18.0, 8], [20.0, 7], [14.0, 11],[8.0, 15], [2.0, 15], [9.0, 12], [14.0, 10], [24.0, 7], [27.0, 13], [17.0, 15], [14.0, 9], [11.0, 5], [7.0, 7], [9.0, 7], [22.0, 6], [19.0, 8], [13.0, 7], [10.0, 6], [13.0, 8], [11.0, 5], [4.0, 7], [6.0, 15], [19.0, 8], [22.0, 8], [27.0, 9], [31.0, 12], [2.0, 15], [6.0, 15], [14.0, 9], [18.0, 10], [21.0, 9], [23.0, 9], [31.0, 13], [18.0, 12], [11.0, 5], [7.0, 11], [27.0, 13], [31.0, 12], [29.0, 7], [20.0, 7], [24.0, 7], [25.0, 8], [0.0, 5], [4.0, 9], [6.0, 10], [14.0, 9], [18.0, 9], [19.0, 10], [17.0, 7], [24.0, 6], [26.0, 12], [6.0, 8], [5.0, 9]])

preds_heatmap[:, 0] = preds_heatmap[:, 0] *(360/32)
preds_heatmap[:, 1] = preds_heatmap[:, 1] *(60.65/16)

preds_rad = np.array([[ 4.3512,  0.6188],
        [ 3.4699,  0.8115],
        [ 1.7628,  0.8110],
        [ 1.8939,  0.7471],
        [ 2.3044,  0.9153],
        [ 1.7799,  0.9464],
        [ 0.8679,  0.9434],
        [ 0.3223,  0.9584],
        [ 1.2365,  0.6117],
        [ 1.5375,  1.0865],
        [ 1.5544,  0.4910],
        [ 3.6171,  0.6699],
        [ 4.7085,  0.7014],
        [ 5.1083,  0.8183],
        [ 5.7473,  0.9197],
        [ 4.1175,  0.6469],
        [ 3.0332,  0.6314],
        [ 1.3408,  0.7343],
        [-0.3703,  1.9112],
        [ 4.6941,  0.8167],
        [ 0.8637,  1.2124],
        [ 0.7779,  0.8639],
        [ 5.0300,  0.8339],
        [ 4.3850,  0.6896],
        [ 5.2336,  0.8409],
        [ 5.6436,  0.4510],
        [ 6.0303,  0.3418],
        [ 3.3696,  0.4989],
        [ 3.1572,  0.7170],
        [ 2.3215,  0.7557],
        [ 0.9270,  0.5469],
        [ 1.8904,  0.5021],
        [ 4.1491,  0.3819],
        [ 4.6514,  0.5308],
        [ 5.9874,  0.3545],
        [ 0.4995,  0.5806],
        [ 3.7992,  0.6018],
        [ 2.7888,  0.5706],
        [ 1.1601,  0.5488],
        [ 1.6439,  0.8201],
        [ 2.3531,  0.8190],
        [ 2.5934,  0.7181],
        [ 5.4074,  0.4028],
        [ 5.6644,  0.5147],
        [ 1.7334,  0.5548],
        [ 1.0420,  0.4946],
        [ 0.7190,  0.6151],
        [ 3.2657,  0.7020],
        [ 3.8452,  0.4087],
        [ 2.6195,  0.8461],
        [ 1.2709,  0.9123],
        [ 0.7463,  0.8399],
        [ 1.4458,  0.6217],
        [ 2.7107,  0.6202],
        [ 5.1128,  0.7445],
        [ 5.5315,  0.7324],
        [ 3.7529,  0.9447],
        [ 3.0332,  0.6052],
        [ 1.9196,  0.4215],
        [ 0.9426,  0.6355],
        [ 0.8487,  0.7624],
        [ 5.4573,  0.3808],
        [ 3.4757,  0.4720],
        [ 2.7158,  0.6171],
        [ 1.6739,  0.5199],
        [ 2.2926,  0.5913],
        [ 1.2273,  0.4878],
        [ 0.9015,  0.5664],
        [ 1.1418,  0.7664],
        [ 3.6653,  0.6106],
        [ 4.5275,  0.7009],
        [ 5.3022,  0.5391],
        [ 5.8455,  0.6726],
        [ 0.5326,  0.6256],
        [ 1.0934,  0.8517],
        [ 2.8868,  0.6625],
        [ 3.6724,  0.7582],
        [ 4.1512,  0.6635],
        [ 4.9637,  0.6602],
        [ 5.8166,  0.7226],
        [ 3.2460,  0.8375],
        [ 2.1047,  0.6255],
        [ 1.8970,  0.8288],
        [ 5.5806,  0.7906],
        [ 5.8046,  0.7445],
        [ 5.6964,  0.7135],
        [ 3.8122,  0.4916],
        [ 5.0970,  0.6289],
        [ 5.4677,  0.5160],
        [ 5.8349,  0.5026],
        [ 0.8484,  0.7676],
        [ 1.3056,  0.7763],
        [ 2.7311,  0.7349],
        [ 3.5055,  0.7033],
        [ 4.1732,  0.6380],
        [ 3.2858,  0.6039],
        [ 5.2122,  0.5746],
        [ 5.4375,  0.7297],
        [ 1.3876,  0.5059],
        [ 1.1479,  0.8062]])


labels = np.array([[3.83729988, 0.59397021],
                        [3.15089407, 0.59346277],
                        [2.91352712, 0.59373891],
                        [2.67859313, 0.59339292],
                        [2.32391487, 0.59362382],
                        [1.48287332, 0.5937624 ],
                        [0.92694105, 0.59392395],
                        [0.54801345, 0.59371653],
                        [2.8306328,  0.82004674],
                        [1.89507113, 0.82230831],
                        [1.90697456, 0.82230831],
                        [3.07727234, 0.71243718],
                        [4.18251045, 0.73480803],
                        [5.11775969, 0.77269811],
                        [6.03934642, 0.62359086],
                        [3.50094072, 0.55520048],
                        [2.92227659, 0.83020729],
                        [1.40277029, 0.7660213 ],
                        [0.87172392, 0.69969345],
                        [4.65316388, 0.73484542],
                        [1.51043012, 0.7256027 ],
                        [0.66392218, 0.72895635],
                        [4.98414982, 0.73127865],
                        [4.06206127, 0.73966369],
                        [5.17382258, 0.73969559],
                        [4.91686482, 0.60351201],
                        [4.38539198, 0.65688627],
                        [3.50326012, 0.60014742],
                        [2.80424665, 0.55286863],
                        [1.92945302, 0.53183852],
                        [0.77022009, 0.62550596],
                        [1.78575238, 0.62550596],
                        [3.84714236, 0.58890528],
                        [4.41877126, 0.56807037],
                        [5.01847162, 0.73049788],
                        [0.29484511, 0.52956178],
                        [3.76136317, 0.45908789],
                        [2.43498889, 0.6345785 ],
                        [1.08762153, 0.61766149],
                        [1.23202621, 0.49430065],
                        [2.14314786, 0.49111503],
                        [2.17441304, 0.57702814],
                        [4.62388944, 0.62928909],
                        [5.53866716, 0.56440384],
                        [1.31286731, 0.60728775],
                        [0.45686295, 0.52875271],
                        [0.45686295, 0.52875271],
                        [3.08940571, 0.56460303],
                        [3.75449916, 0.60358645],
                        [2.47339746, 0.69489381],
                        [1.50718856, 0.76397814],
                        [0.38902807, 0.73334943],
                        [1.63836546, 0.78521541],
                        [2.51419918, 0.7887945 ],
                        [4.587151  , 0.90069939],
                        [5.19844243, 0.95400255],
                        [3.20934968, 0.67794078],
                        [2.6550299 , 0.5413819 ],
                        [1.56264433, 0.62592756],
                        [0.88465583, 0.50352831],
                        [0.4655539 , 0.52413309],
                        [4.26659364, 0.60246085],
                        [3.51824086, 0.57061866],
                        [2.48990163, 0.52557202],
                        [1.65625957, 0.58918089],
                        [2.5179671 , 0.5869228 ],
                        [1.81851289, 0.69651115],
                        [0.68232254, 0.50804771],
                        [1.31833207, 0.649543  ],
                        [3.81632523, 0.63806723],
                        [4.57842669, 0.63990222],
                        [5.3110536 , 0.73602257],
                        [5.98382388, 0.65053742],
                        [0.32787566, 0.63809092],
                        [1.29456344, 0.62563868],
                        [2.90288476, 0.74119749],
                        [3.66509863, 0.66371935],
                        [4.28412167, 0.60464831],
                        [4.99642274, 0.69300019],
                        [5.93242043, 0.61032183],
                        [3.31530819, 0.59304285],
                        [2.19167743, 0.65429815],
                        [1.6145404 , 0.72206872],
                        [5.51693092, 0.62878666],
                        [5.99086293, 0.67453473],
                        [5.99331398, 0.67274036],
                        [3.51346683, 0.75122859],
                        [4.54327391, 0.77926439],
                        [5.10875061, 0.79673333],
                        [5.69237445, 0.88095282],
                        [0.64280898, 0.63873428],
                        [1.30979571, 0.7112817 ],
                        [2.45948609, 0.73671796],
                        [3.26790009, 0.74317413],
                        [3.81794791, 0.78280636],
                        [2.96871366, 0.78278828],
                        [4.53140166, 0.76163143],
                        [5.30097843, 0.65676378],
                        [0.90047127, 0.6811048 ],
                        [0.92654884 ,0.70376786]])

preds_rad = (preds_rad / (np.pi*2)) * 360
labels = (labels / (np.pi*2)) * 360

preds_da = np.array([[22, 11], [17, 11], [14, 15], [11, 14], [11, 15], [6, 15], [3, 15], [30, 15], [6, 15], [9, 15], [3, 15], [17, 15], [25, 13], [27, 15], [27, 15], [19, 14], [17, 15], [6, 13], [6, 15], [22, 15], [6, 14], [3, 15], [25, 15], [22, 15], [25, 15], [30, 3], [30, 13], [17, 13], [14, 13], [11, 13], [6, 12], [6, 11], [19, 11], [22, 13], [27, 11], [3, 13], [17, 14], [14, 13], [6, 10], [9, 13], [11, 13], [14, 13], [25, 13], [27, 10], [9, 13], [3, 13], [3, 13], [17, 13], [19, 11], [14, 14], [6, 15], [3, 14], [6, 13], [14, 14], [25, 14], [25, 12], [17, 14], [14, 13], [6, 12], [3, 10], [3, 12], [25, 13], [17, 13], [14, 13], [6, 13], [11, 9], [6, 7], [3, 12], [6, 14], [17, 13], [22, 11], [25, 10], [30, 13], [1, 13], [6, 13], [14, 13], [17, 13], [19, 13], [22, 13], [27, 13], [17, 14], [11, 14], [9, 14], [27, 13], [27, 14], [27, 13], [19, 13], [25, 13], [27, 12], [27, 11], [3, 12], [6, 13], [14, 13], [17, 12], [17, 13], [14, 13], [27, 13], [27, 13], [6, 12], [6, 13]])

preds_da[:, 0] = preds_da[:, 0] *(360/32)
preds_da[:, 1] = preds_da[:, 1] *(60.65/16)



def visualize(preds, labels):
    count10 = 0
    count15 = 0
    count20 = 0
    fig, ax = plt.subplots()
    for pred, label in zip(preds, labels):
        error_alpha = pred[0] - label[0]
        error_beta = pred[1] - label[1]
        ax.scatter(error_alpha, error_beta, color='blue')
        if abs(error_alpha) <= 10 and abs(error_beta) <= 10:
            count10 += 1
        if abs(error_alpha) <= 15 and abs(error_beta) <= 15:
            count15 += 1
        if abs(error_alpha) <= 20 and abs(error_beta) <= 20:
            count20 += 1
    print(count10)
    #print(count15)
    print(count20)
    circle10 = Circle((0, 0), 10, color='#76c68f', fill=True, alpha=0.7, lw=2) #hatch='xxxxx'
    #circle15 = Circle((0, 0), 15, color='green', fill=False, alpha=0.5, hatch='|', lw=2)
    circle20 = Circle((0, 0), 20, color='#c86558', fill=True, alpha=0.3, lw=2)
    ax.add_patch(circle10)
    #ax.add_patch(circle15)

    line20, = ax.plot([0, 0], [0, 0], color='#c86558', lw=6, label='< 20', alpha=0.5)
    line10, = ax.plot([0, 0], [0, 0], color='#76c68f', lw=6, label='< 10', alpha=0.7)

    ax.add_patch(circle20)
    ax.set_xlabel('Napaka alpha(°)')
    ax.set_ylabel('Napaka beta(°)')
    ax.scatter(0, 0, color='red', zorder=10)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.legend([line20, line10], [f'α,β < 20\n{count20}% primerov', f'α,β < 10\n{count10}% primerov'], loc='upper right')
    ax.grid(True, linestyle='--', color='gray', linewidth=0.5, zorder=0)
    plt.show()


def visualize_alpha(preds, labels):
    fig, ax = plt.subplots()
    for pred, label in zip(preds, labels):
        error_alpha = abs(pred[0] - label[0])
        ax.scatter(label[0], error_alpha, color='blue')
    mean_error = np.mean(abs(preds[:, 0] - labels[:, 0]))
    ax.set_xlabel('Kot alpha')
    ax.set_ylabel('Napaka kota alpha')
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 100)
    ax.plot([10, 350], [mean_error, mean_error], color='red', lw=2, linestyle='--')
    plt.show()

def visualize_beta(preds, labels):
    fig, ax = plt.subplots()
    for pred, label in zip(preds, labels):
        error_beta = abs(pred[1] - label[1])
        ax.scatter(label[1], error_beta, color='green')
    mean_error = np.mean(abs(preds[:, 1] - labels[:, 1]))
    ax.set_xlabel('Kot beta')
    ax.set_ylabel('Napaka kota beta')
    ax.set_xlim(25, 60)
    ax.set_ylim(0, 60)
    ax.plot([30, 55], [mean_error, mean_error], color='red', lw=2, linestyle='--')
    plt.show()


if __name__ == '__main__':
    visualize(preds_heatmap, labels)
    #visualize_alpha(preds, labels)
    #visualize_beta(preds, labels)
    visualize(preds_rad, labels)
    #visualize(preds_da, labels)