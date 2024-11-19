import matplotlib.pyplot as plt
import numpy as np

loss1 = np.array([1.2714646875858306, 0.7652614542841911, 0.5866450682282448, 0.43217074424028395, 0.41105042278766635, 0.35144872568547725, 0.2958396475017071, 0.2604203573614359, 0.2384389128163457, 0.17395753528922797, 0.2097520576044917, 0.15816899409517646, 0.1372792195621878, 0.13891721686348318, 0.14818380361422898, 0.1023069712985307, 0.12786008323542775, 0.11265970576088875, 0.11918496409896762, 0.08438062475528568, 0.0965678876126185, 0.07314035410061479, 0.07770831960253417, 0.06832518665352837, 0.08051114362198859, 0.06962014329386874, 0.06008031902601942, 0.05664828127017245, 0.06773574075195939, 0.059897698753047734, 0.045636341467034075, 0.07264887273311615, 0.07354158980306238, 0.06154416055418551, 0.04854507731506601, 0.056236719405278565, 0.03938840896356851, 0.047821898988913744, 0.06249862600350752, 0.0502623159927316, 0.051390594763215634, 0.03465462631429546])
vloss1 = np.array([0.9922948684332505, 0.6466031341057903, 0.5403238432025009, 0.3871305976836186, 0.36943306832943323, 0.2652666363232541, 0.2441130409684946, 0.18055838368973642, 0.4821461209429885, 0.11932422629083103, 0.22918584976556167, 0.12983063771125844, 0.13810017069555677, 0.0924985645760624, 0.12840953133648858, 0.08111689696264154, 0.05375347275919509, 0.1356342336198069, 0.06134980419510097, 0.07216008949392247, 0.0947967525053207, 0.1250476548439419, 0.057190860836889665, 0.04984081979261114, 0.0543083191243053, 0.12709365444222712, 0.05134027022429092, 0.05694487122291664, 0.05096711467705527, 0.043414521288952596, 0.08760346288019616, 0.029285418958280164, 0.06224367133394446, 0.05927829754117103, 0.15631053896515437, 0.035214045172873534, 0.030500723210049197, 0.05025375439381262, 0.03517922302312657, 0.05009698570455547, 0.055729460326585704, 0.0403150405281197])

epochs1 = np.arange(1, len(loss1) + 1, 1)

plt.suptitle('Loss history visualization', fontsize=16)

plt.subplot(1, 3, 1)
plt.title("efficient_net_b3_discrete")

plt.plot(epochs1, loss1, 'r', label="loss")
plt.plot(epochs1, vloss1, 'b', label="validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right")
#plt.ylim(12, 15)

loss2 = np.array([1.567932928800583, 0.9480510747432709, 0.7361367231607437, 0.5747219559550285, 0.5041813050210476, 0.38673782475292684, 0.33967766754329204, 0.31557595245540143, 0.31829030700027944, 0.22356792360544206, 0.22920978017151356, 0.2265625186264515, 0.15107456553727389, 0.18624936213716864, 0.1581280973739922, 0.1418311226926744, 0.13473323789425196, 0.1575489601586014, 0.11286960147321225, 0.12624604668468237, 0.1528271242044866, 0.0803964624274522, 0.11473946297541261, 0.07548303166870028, 0.09128307906910776, 0.11239610970485955, 0.08521730814361944, 0.07791383775649592, 0.06658690579468385, 0.07391462026163936, 0.08888604836771265, 0.038708510827273133, 0.05687392829917371, 0.08534149248851464, 0.05342415672261268, 0.06582849404308945, 0.06163795077474788, 0.03585671255481429])
vloss2 = np.array([1.4757713248144906, 0.7815923126238697, 0.9240266180263376, 0.8763867514313392, 0.8738137366636745, 0.6272590452207709, 0.665374874956203, 0.4181669857704414, 0.6628026856566375, 0.2761476270011011, 0.3442657753825188, 0.11966194428925245, 0.10467645098578256, 0.6140138275499614, 0.1293077319227862, 0.13955505279788993, 0.11514046560832351, 0.06706228448816065, 0.09813281982117948, 0.05866596293723527, 0.2977067670169866, 0.061227626166359156, 0.07076117963055675, 0.325060680909258, 0.3736840726456271, 0.4241511132286967, 0.06098712873557266, 0.034213195256625285, 0.037333037439329586, 0.13580616745254342, 0.31657463064733543, 0.039219647967959494, 0.06564812407969445, 0.10129333327761587, 0.06023383964169419, 0.03531888030372771, 0.043107470071364964, 0.08643512462775381])

epochs2 = np.arange(1, len(loss2) + 1, 1)

plt.subplot(1, 3, 2)
plt.title("efficient_net_b5_discrete")

plt.plot(epochs2, loss2, 'r', label="loss")
plt.plot(epochs2, vloss2, 'b', label="validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend(loc="upper right")

loss3 = np.array([1.7580513417720796, 0.8930146068334579, 0.7006121385097503, 0.5105264657735824, 0.44023836150765416, 0.3666310876607895, 0.3283202857524157, 0.29007851868867873, 0.29211510334163904, 0.2407619021832943, 0.21001800023019312, 0.18889070473611355, 0.22005066122859718, 0.16281262323260307, 0.16465026788413525, 0.16424162616953253, 0.11603293221443892, 0.14196157502941786, 0.13622407210990786, 0.09948809517547488, 0.1354202430136502, 0.09064560481347143, 0.10512156972661614, 0.09414817155804485, 0.10982239826582373, 0.1145160104893148, 0.06509956445544958, 0.08010960605926812, 0.08064980161143467, 0.09189996609464288, 0.09029806197620928, 0.08867288752458989, 0.07739420135971159, 0.04911102145910263, 0.07063745269319043, 0.05811782406177372, 0.0832744358573109, 0.054895199295133354, 0.062184676560573277, 0.08420529258903116, 0.06346225695917383, 0.07145852796500549, 0.06328019007109105, 0.044074456421658394, 0.047976413738215345, 0.058959122837986794, 0.05686242440948263, 0.050603576807770875, 0.06837493491533678, 0.044019755320623515, 0.04586202509468421, 0.051847914935788136, 0.07352775409119204, 0.04698678941000253, 0.03640229807817377, 0.040977403445867824, 0.04831736807129346, 0.0429240611451678, 0.0429823243978899, 0.038941226778551934, 0.04227851625182666, 0.053361040097661315, 0.03416983604605775, 0.035305732546839864, 0.04477535131853074, 0.04853127612615935, 0.03257140701869503, 0.03971691672923043, 0.05250771307677496, 0.03501504885382019, 0.044179471615934746, 0.03712431697291322, 0.03961710251809564, 0.035417481350013984, 0.02540236853703391, 0.029849909623735585])
vloss3 = np.array([1.3390767650784186, 0.6950325496916501, 0.5960902688638219, 0.598878346411687, 0.3402563823843902, 0.2886223427529605, 0.23566945515713603, 0.2221009056523161, 0.21128953473185594, 0.18822791260087265, 0.20536012919443958, 0.12890625182047205, 0.204840603112331, 0.11462550192965933, 0.14771116316177935, 0.1301165133890397, 0.08780744247454798, 0.15688630215964228, 0.08526248613477878, 0.0725462426141537, 0.056558870532954074, 0.22639321563589684, 0.1740524232844418, 0.044651849094321426, 0.06527227757691915, 0.07546713330150354, 0.1347499978544085, 0.06536502985213444, 0.046696236197625834, 0.07089112719591215, 0.05634345824496364, 0.04146737917414251, 0.07996602663067433, 0.1350400676621335, 0.03655480196225053, 0.04458955285438108, 0.03659090428033246, 0.056286144416498124, 0.07237943764490444, 0.04103502582117283, 0.04944812878878471, 0.043583354809681694, 0.03127291046394879, 0.022940630572057277, 0.0823807774753011, 0.022343166294191384, 0.14229202338599795, 0.04302610828396249, 0.027952698144604857, 0.04079518553115568, 0.05919229011292974, 0.04150671170835541, 0.018392832769204, 0.031055864002028446, 0.03786527180345729, 0.025149292727923548, 0.029250812222785756, 0.057319513320448406, 0.01343391634884097, 0.03448423921165742, 0.05063271122053266, 0.016998860135428468, 0.045101001901762944, 0.04659841713420871, 0.02503754966507912, 0.009470668066850917, 0.02766975507904547, 0.029037967353360727, 0.030528984918206367, 0.021093580686873843, 0.027530239994707956, 0.019259525324714387, 0.022231070925635373, 0.02294961469366549, 0.026949387161667407, 0.04059677241229505])

epochs3 = np.arange(1, len(loss3) + 1, 1)

plt.subplot(1, 3, 3)
plt.title("efficient_net_b7_discrete")

plt.plot(epochs3, loss3, 'r', label="loss")
plt.plot(epochs3, vloss3, 'b', label="validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend(loc="upper right")

plt.show()
