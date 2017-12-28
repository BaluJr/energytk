import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
#import seaborn as sns

class SlpForecasterModel(object):
    '''
    This model class contains the slp models which are used for the calculation.
    
    Attributes
    ----------
    Contains the SLPs and the definition of the seasons.
    '''

    params = {
    }


    # The Default H0 SLPs 
    WEEK_WINTER = [ 0.0676, 0.0608, 0.0549, 0.0499, 0.0462, 0.0436, 0.0419, 0.0408, 0.0401, 0.0396, 0.0394, 0.0391, 0.0388, 0.0386, 0.0383, 0.0383, 0.0384, 0.0388, 0.0393, 0.0400, 0.0409, 0.0431, 0.0477, 0.0558, 0.0680, 0.0828, 0.0980, 0.1115, 0.1216, 0.1285, 0.1327, 0.1348, 0.1354, 0.1348, 0.1331, 0.1307, 0.1277, 0.1246, 0.1215, 0.1190, 0.1173, 0.1162, 0.1157, 0.1157, 0.1161, 0.1170, 0.1187, 0.1215, 0.1254, 0.1296, 0.1330, 0.1348, 0.1342, 0.1317, 0.1280, 0.1240, 0.1202, 0.1168, 0.1137, 0.1107, 0.1079, 0.1055, 0.1035, 0.1024, 0.1022, 0.1032, 0.1056, 0.1099, 0.1160, 0.1237, 0.1326, 0.1423, 0.1524, 0.1622, 0.1712, 0.1789, 0.1847, 0.1882, 0.1889, 0.1864, 0.1807, 0.1727, 0.1639, 0.1556, 0.1489, 0.1434, 0.1384, 0.1332, 0.1272, 0.1205, 0.1133, 0.1057, 0.0980, 0.0902, 0.0825, 0.0749 ]
    SAT_WINTER = [ 0.0708, 0.0682, 0.0659, 0.0633, 0.0595, 0.0550, 0.0505, 0.0466, 0.0439, 0.0423, 0.0414, 0.0408, 0.0403, 0.0399, 0.0395, 0.0391, 0.0388, 0.0385, 0.0383, 0.0383, 0.0385, 0.0391, 0.0403, 0.0424, 0.0456, 0.0499, 0.0553, 0.0616, 0.0689, 0.0771, 0.0861, 0.0957, 0.1058, 0.1158, 0.1249, 0.1323, 0.1376, 0.1411, 0.1433, 0.1448, 0.1460, 0.1472, 0.1484, 0.1498, 0.1515, 0.1535, 0.1560, 0.1590, 0.1624, 0.1658, 0.1684, 0.1698, 0.1694, 0.1676, 0.1648, 0.1615, 0.1581, 0.1549, 0.1518, 0.1490, 0.1465, 0.1444, 0.1427, 0.1415, 0.1409, 0.1417, 0.1449, 0.1515, 0.1619, 0.1746, 0.1874, 0.1981, 0.2052, 0.2091, 0.2111, 0.2122, 0.2132, 0.2130, 0.2104, 0.2039, 0.1929, 0.1790, 0.1644, 0.1515, 0.1419, 0.1353, 0.1310, 0.1282, 0.1261, 0.1241, 0.1216, 0.1182, 0.1134, 0.1074, 0.1008, 0.0941 ]
    SUN_WINTER = [ 0.0875, 0.0811, 0.0750, 0.0691, 0.0634, 0.0582, 0.0536, 0.0499, 0.0473, 0.0455, 0.0442, 0.0433, 0.0424, 0.0415, 0.0407, 0.0400, 0.0393, 0.0388, 0.0385, 0.0383, 0.0383, 0.0384, 0.0387, 0.0391, 0.0397, 0.0404, 0.0413, 0.0424, 0.0440, 0.0466, 0.0511, 0.0583, 0.0686, 0.0813, 0.0952, 0.1090, 0.1219, 0.1337, 0.1444, 0.1540, 0.1626, 0.1705, 0.1780, 0.1856, 0.1933, 0.2006, 0.2066, 0.2106, 0.2118, 0.2102, 0.2059, 0.1989, 0.1896, 0.1787, 0.1673, 0.1565, 0.1470, 0.1389, 0.1321, 0.1265, 0.1220, 0.1181, 0.1148, 0.1115, 0.1083, 0.1059, 0.1052, 0.1074, 0.1129, 0.1209, 0.1300, 0.1390, 0.1468, 0.1535, 0.1597, 0.1656, 0.1715, 0.1764, 0.1791, 0.1781, 0.1729, 0.1647, 0.1556, 0.1473, 0.1414, 0.1372, 0.1337, 0.1298, 0.1248, 0.1186, 0.1116, 0.1040, 0.0962, 0.0884, 0.0807, 0.0732]
    WEEK_SUMMER = [ 0.0863, 0.0769, 0.0688, 0.0624, 0.0580, 0.0553, 0.0536, 0.0524, 0.0513, 0.0503, 0.0492, 0.0483, 0.0475, 0.0469, 0.0465, 0.0466, 0.0471, 0.0480, 0.0493, 0.0508, 0.0527, 0.0556, 0.0605, 0.0682, 0.0792, 0.0920, 0.1047, 0.1157, 0.1235, 0.1286, 0.1320, 0.1348, 0.1378, 0.1407, 0.1432, 0.1448, 0.1453, 0.1449, 0.1438, 0.1423, 0.1408, 0.1395, 0.1385, 0.1382, 0.1386, 0.1401, 0.1426, 0.1465, 0.1515, 0.1567, 0.1607, 0.1623, 0.1605, 0.1561, 0.1502, 0.1440, 0.1384, 0.1336, 0.1294, 0.1257, 0.1224, 0.1196, 0.1174, 0.1157, 0.1146, 0.1142, 0.1146, 0.1157, 0.1176, 0.1203, 0.1239, 0.1282, 0.1332, 0.1389, 0.1451, 0.1515, 0.1579, 0.1638, 0.1683, 0.1706, 0.1704, 0.1683, 0.1653, 0.1623, 0.1601, 0.1584, 0.1568, 0.1548, 0.1519, 0.1479, 0.1425, 0.1357, 0.1272, 0.1175, 0.1071, 0.0965 ]
    SAT_SUMMER = [ 0.0898, 0.0849, 0.0807, 0.0766, 0.0717, 0.0666, 0.0616, 0.0574, 0.0545, 0.0526, 0.0514, 0.0508, 0.0503, 0.0500, 0.0499, 0.0499, 0.0501, 0.0504, 0.0507, 0.0508, 0.0508, 0.0509, 0.0516, 0.0533, 0.0562, 0.0604, 0.0658, 0.0724, 0.0800, 0.0885, 0.0974, 0.1065, 0.1156, 0.1244, 0.1328, 0.1407, 0.1478, 0.1540, 0.1589, 0.1623, 0.1641, 0.1647, 0.1650, 0.1656, 0.1671, 0.1694, 0.1724, 0.1756, 0.1790, 0.1819, 0.1837, 0.1839, 0.1821, 0.1787, 0.1741, 0.1690, 0.1637, 0.1589, 0.1546, 0.1515, 0.1496, 0.1486, 0.1480, 0.1473, 0.1462, 0.1450, 0.1441, 0.1440, 0.1449, 0.1469, 0.1500, 0.1540, 0.1588, 0.1640, 0.1692, 0.1740, 0.1778, 0.1805, 0.1818, 0.1814, 0.1793, 0.1757, 0.1710, 0.1656, 0.1601, 0.1551, 0.1511, 0.1490, 0.1489, 0.1496, 0.1494, 0.1465, 0.1398, 0.1303, 0.1195, 0.1090 ]
    SUN_SUMMER = [ 0.1001, 0.0925, 0.0859, 0.0799, 0.0741, 0.0687, 0.0639, 0.0599, 0.0570, 0.0550, 0.0535, 0.0524, 0.0515, 0.0508, 0.0502, 0.0499, 0.0499, 0.0500, 0.0501, 0.0499, 0.0495, 0.0489, 0.0484, 0.0483, 0.0487, 0.0498, 0.0519, 0.0549, 0.0592, 0.0649, 0.0723, 0.0816, 0.0929, 0.1056, 0.1190, 0.1323, 0.1448, 0.1562, 0.1660, 0.1740, 0.1800, 0.1846, 0.1887, 0.1931, 0.1983, 0.2037, 0.2087, 0.2122, 0.2137, 0.2128, 0.2090, 0.2023, 0.1924, 0.1809, 0.1692, 0.1590, 0.1514, 0.1460, 0.1416, 0.1373, 0.1324, 0.1270, 0.1216, 0.1165, 0.1122, 0.1089, 0.1067, 0.1057, 0.1061, 0.1077, 0.1104, 0.1140, 0.1183, 0.1234, 0.1292, 0.1357, 0.1428, 0.1498, 0.1557, 0.1598, 0.1614, 0.1608, 0.1590, 0.1565, 0.1539, 0.1515, 0.1493, 0.1473, 0.1454, 0.1430, 0.1392, 0.1332, 0.1244, 0.1138, 0.1025, 0.0916 ]
    WEEK_INTER = [0.0778, 0.0696, 0.0624, 0.0566, 0.0525, 0.0497, 0.0479, 0.0466, 0.0455, 0.0445, 0.0438, 0.0433, 0.0430, 0.0430, 0.0431, 0.0433, 0.0434, 0.0437, 0.0442, 0.0449, 0.0463, 0.0489, 0.0537, 0.0616, 0.0729, 0.0863, 0.1001, 0.1124, 0.1218, 0.1285, 0.1329, 0.1357, 0.1372, 0.1377, 0.1377, 0.1373, 0.1369, 0.1364, 0.1357, 0.1348, 0.1337, 0.1324, 0.1314, 0.1307, 0.1306, 0.1315, 0.1336, 0.1373, 0.1426, 0.1482, 0.1528, 0.1548, 0.1532, 0.1489, 0.1432, 0.1373, 0.1324, 0.1284, 0.1248, 0.1215, 0.1181, 0.1148, 0.1117, 0.1090, 0.1069, 0.1057, 0.1055, 0.1065, 0.1091, 0.1131, 0.1183, 0.1248, 0.1324, 0.1406, 0.1491, 0.1573, 0.1649, 0.1711, 0.1752, 0.1765, 0.1745, 0.1705, 0.1657, 0.1615, 0.1589, 0.1572, 0.1554, 0.1523, 0.1472, 0.1403, 0.1321, 0.1232, 0.1140, 0.1048, 0.0956, 0.0866 ]
    SAT_INTER = [0.0802, 0.0751, 0.0707, 0.0666, 0.0623, 0.0580, 0.0541, 0.0508, 0.0484, 0.0468, 0.0457, 0.0449, 0.0444, 0.0439, 0.0435, 0.0433, 0.0431, 0.0431, 0.0431, 0.0433, 0.0436, 0.0442, 0.0454, 0.0474, 0.0505, 0.0549, 0.0607, 0.0682, 0.0775, 0.0879, 0.0986, 0.1090, 0.1184, 0.1267, 0.1338, 0.1398, 0.1447, 0.1488, 0.1524, 0.1556, 0.1589, 0.1620, 0.1649, 0.1673, 0.1692, 0.1708, 0.1725, 0.1748, 0.1777, 0.1808, 0.1831, 0.1839, 0.1827, 0.1798, 0.1758, 0.1715, 0.1672, 0.1631, 0.1592, 0.1556, 0.1523, 0.1495, 0.1472, 0.1457, 0.1450, 0.1454, 0.1467, 0.1490, 0.1523, 0.1564, 0.1615, 0.1673, 0.1737, 0.1804, 0.1867, 0.1923, 0.1965, 0.1990, 0.1994, 0.1973, 0.1924, 0.1852, 0.1766, 0.1673, 0.1579, 0.1495, 0.1429, 0.1390, 0.1383, 0.1392, 0.1395, 0.1373, 0.1311, 0.1219, 0.1115, 0.1015 ]
    SUN_INTER = [0.0934, 0.0868, 0.0812, 0.0757, 0.0701, 0.0645, 0.0593, 0.0549, 0.0517, 0.0494, 0.0478, 0.0466, 0.0455, 0.0445, 0.0438, 0.0433, 0.0431, 0.0431, 0.0432, 0.0433, 0.0433, 0.0433, 0.0432, 0.0433, 0.0435, 0.0443, 0.0460, 0.0491, 0.0539, 0.0604, 0.0688, 0.0791, 0.0911, 0.1043, 0.1180, 0.1315, 0.1442, 0.1555, 0.1653, 0.1731, 0.1788, 0.1831, 0.1870, 0.1914, 0.1970, 0.2030, 0.2085, 0.2122, 0.2135, 0.2118, 0.2070, 0.1989, 0.1876, 0.1746, 0.1617, 0.1506, 0.1428, 0.1376, 0.1339, 0.1307, 0.1271, 0.1231, 0.1190, 0.1149, 0.1110, 0.1077, 0.1056, 0.1049, 0.1060, 0.1088, 0.1132, 0.1190, 0.1260, 0.1337, 0.1415, 0.1490, 0.1555, 0.1606, 0.1636, 0.1640, 0.1615, 0.1571, 0.1522, 0.1482, 0.1459, 0.1447, 0.1436, 0.1415, 0.1375, 0.1318, 0.1247, 0.1165, 0.1076, 0.0984, 0.0892, 0.0807 ]
    DAY_FACTORS = [1.24203, 1.24392, 1.24568, 1.24730, 1.24878, 1.25014, 1.25137, 1.25247, 1.25344, 1.25430, 1.25503, 1.25564, 1.25613, 1.25650, 1.25677, 1.25691, 1.25695, 1.25688, 1.25670, 1.25642, 1.25603, 1.25554, 1.25495, 1.25426, 1.25347, 1.25259, 1.25161, 1.25055, 1.24939, 1.24814, 1.24681, 1.24539, 1.24389, 1.24230, 1.24064, 1.23889, 1.23707, 1.23517, 1.23320, 1.23116, 1.22904, 1.22686, 1.22460, 1.22228, 1.21990, 1.21745, 1.21494, 1.21237, 1.20974, 1.20705, 1.20431, 1.20151, 1.19866, 1.19575, 1.19280, 1.18979, 1.18674, 1.18365, 1.18051, 1.17732, 1.17409, 1.17082, 1.16752, 1.16417, 1.16079, 1.15737, 1.15392, 1.15043, 1.14692, 1.14337, 1.13979, 1.13619, 1.13256, 1.12890, 1.12522, 1.12152, 1.11779, 1.11405, 1.11029, 1.10650, 1.10270, 1.09889, 1.09506, 1.09122, 1.08736, 1.08350, 1.07962, 1.07573, 1.07184, 1.06794, 1.06404, 1.06012, 1.05621, 1.05229, 1.04838, 1.04446, 1.04054, 1.03662, 1.03271, 1.02880, 1.02489, 1.02099, 1.01710, 1.01321, 1.00934, 1.00547, 1.00161, 0.99776, 0.99393, 0.99011, 0.98630, 0.98251, 0.97873, 0.97497, 0.97122, 0.96750, 0.96379, 0.96011, 0.95644, 0.95279, 0.94917, 0.94557, 0.94200, 0.93845, 0.93492, 0.93142, 0.92795, 0.92451, 0.92109, 0.91770, 0.91434, 0.91102, 0.90772, 0.90445, 0.90122, 0.89802, 0.89486, 0.89173, 0.88863, 0.88557, 0.88254, 0.87956, 0.87661, 0.87369, 0.87082, 0.86799, 0.86519, 0.86244, 0.85972, 0.85705, 0.85442, 0.85183, 0.84928, 0.84678, 0.84432, 0.84191, 0.83954, 0.83721, 0.83493, 0.83270, 0.83051, 0.82837, 0.82628, 0.82423, 0.82223, 0.82028, 0.81838, 0.81653, 0.81473, 0.81298, 0.81128, 0.80962, 0.80802, 0.80647, 0.80497, 0.80352, 0.80213, 0.80078, 0.79949, 0.79825, 0.79707, 0.79593, 0.79485, 0.79383, 0.79286, 0.79194, 0.79107, 0.79026, 0.78950, 0.78880, 0.78815, 0.78756, 0.78702, 0.78654, 0.78611, 0.78574, 0.78542, 0.78516, 0.78495, 0.78480, 0.78470, 0.78466, 0.78468, 0.78475, 0.78487, 0.78505, 0.78529, 0.78558, 0.78593, 0.78633, 0.78679, 0.78731, 0.78788, 0.78850, 0.78918, 0.78991, 0.79070, 0.79155, 0.79244, 0.79340, 0.79440, 0.79546, 0.79658, 0.79775, 0.79897, 0.80025, 0.80158, 0.80296, 0.80440, 0.80588, 0.80742, 0.80902, 0.81066, 0.81236, 0.81410, 0.81590, 0.81775, 0.81965, 0.82160, 0.82360, 0.82565, 0.82774, 0.82989, 0.83209, 0.83433, 0.83662, 0.83896, 0.84134, 0.84377, 0.84625, 0.84877, 0.85134, 0.85395, 0.85661, 0.85931, 0.86205, 0.86484, 0.86767, 0.87054, 0.87345, 0.87641, 0.87940, 0.88243, 0.88551, 0.88862, 0.89177, 0.89495, 0.89818, 0.90144, 0.90473, 0.90806, 0.91143, 0.91483, 0.91826, 0.92172, 0.92522, 0.92875, 0.93230, 0.93589, 0.93951, 0.94315, 0.94683, 0.95053, 0.95425, 0.95800, 0.96178, 0.96558, 0.96941, 0.97325, 0.97712, 0.98101, 0.98492, 0.98884, 0.99279, 0.99675, 1.00073, 1.00473, 1.00874, 1.01276, 1.01680, 1.02085, 1.02491, 1.02898, 1.03306, 1.03715, 1.04125, 1.04536, 1.04947, 1.05358, 1.05770, 1.06182, 1.06594, 1.07006, 1.07419, 1.07831, 1.08243, 1.08655, 1.09066, 1.09476, 1.09886, 1.10295, 1.10704, 1.11111, 1.11517, 1.11922, 1.12326, 1.12728, 1.13129, 1.13528, 1.13925, 1.14320, 1.14714, 1.15105, 1.15494, 1.15880, 1.16264, 1.16645, 1.17024, 1.17400, 1.17772, 1.18142, 1.18508, 1.18871, 1.19230, 1.19586, 1.19938, 1.20286, 1.20630, 1.20970, 1.21305, 1.21636, 1.21962, 1.22284, 1.22601, 1.22912, 1.23219, 1.23520, 1.23816, 1.24106, 1.24391, 1.24669, 1.24942, 1.25208, 1.25468, 1.25722, 1.25969]


    # The days when to use which load profile
    MID_WINTER = 21;     # 21st jan
    MID_SPRING = 111;    # 20/21st apr
    MID_SUMMER = 202;    # 20/21st july
    MID_FALL = 294;      # 20/21st okt


class SlpForecaster(object):
    """ Forecastr based on the official standard H0 load profiles.
    This forecaster works the same as the grid operators. It takes each building and applies a 
    standard load profiles. Currently the default H0 profile for domestic homes is used.

    Attributes
    ----------
    verbose: bool
        Wheteher additional output shall be given during any function.
    """

    Requirement = {'building_type':'ANY VALUE'}

    model_class = SlpForecasterModel

    def __init__(self, model = None, verbose = False):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it createsa default one.
        
        Paramters
        ---------
        model: Model of type model_class
            The model which shall be used.
        """

        if model == None:
            model = self.model_class();
        self.model = m = model;
        super(SlpForecaster, self).__init__()

        # Load SLPs
        arrays = [np.array(['Summer', 'Summer', 'Summer', 'Winter', 'Winter', 'Winter', 'Inter', 'Inter', 'Inter']), 
                  np.array(['Weekday', 'Saturday', 'Sunday', 'Weekday', 'Saturday', 'Sunday', 'Weekday', 'Saturday', 'Sunday'])]
        tuples = list(zip(*arrays))
        col = pd.MultiIndex.from_tuples(tuples, names=['Season', 'Day'])
        idx = pd.DatetimeIndex(start="1.1.2017", freq='15min', periods=96)
        self.profiles = pd.DataFrame(np.array([m.WEEK_SUMMER, m.SAT_SUMMER, m.SUN_SUMMER, m.WEEK_WINTER, m.SAT_WINTER, m.SUN_WINTER, m.WEEK_INTER, m.SAT_INTER, m.SUN_INTER]).T, columns=col, index = idx)
        self.verbose = verbose

    def plot(self):
        ''' Plots the used standard load profiles. 
        '''
        sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 3})
        ax = profiles['Winter'].plot()
        ax.set_ylabel('Factor')
        ax.set_xlabel('Time of Day')
        ax.yaxis.labelpad = 10
        ax = profiles['Summer'].plot()
        ax.set_ylabel('Factor')
        ax.set_xlabel('Time of Day')
        ax.yaxis.labelpad = 10
        plt.gca().xaxis.set_major_locator(dates.HourLocator())
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
        i = 1



    
    def _forecast_single(self, annual_power, timestamp, horizon = None, resolution = "15m"):
        '''
        Helpfunction encapsulating the forecast of a single timestamp 
        with horizon.
        
        Parameters
        ----------
        annual_power: 
            The annual_power of the household timeline to predict in KWh/a
        timestamp: pd.TimeStamp,...
            The point for which a prognoses shall be done.
        horizon: pd.Timedelta
            If set, the amount of points into the future starting 
            from each timestamp in timestamps.
        resolution: str
            A freq_str representing the frequency with which results
            are returned.

        Returns
        -------
        forecast: pd.Series
            The produced forecast in kW for the horizont after the timestamp
            in the defined resolution.
        '''
        m = self.model
        profiles = self.profiles 
        predictions = pd.Series()
        timestamps = pd.date_range(timestamp, timestamp + horizon, freq=resolution)
        n = len(timestamps)
        for i, day in enumerate(timestamps):
            if self.verbose and i % 10 == 0:
                print("Forecast: {0}/{1}".format(i,n))

            # Calculate the season interpolation factors
            dayOfYear = day.dayofyear
            if m.MID_WINTER <= dayOfYear < m.MID_SPRING:
                shareSummer = 0.0
                shareWinter = 1.0 - (dayOfYear - m.MID_WINTER) / (m.MID_SPRING - m.MID_WINTER)
                shareInter = 1.0 - shareWinter
            elif m.MID_SPRING <= dayOfYear < m.MID_SUMMER:
                shareWinter = 0.0
                shareInter = 1.0 - (dayOfYear - m.MID_SPRING) / (m.MID_SUMMER - m.MID_SPRING)
                shareSummer = 1.0 - shareInter
            elif m.MID_SUMMER <= dayOfYear < m.MID_FALL:
                shareWinter = 0.0
                shareSummer = 1.0 - (dayOfYear - m.MID_SUMMER) / (m.MID_FALL - m.MID_SUMMER)
                shareInter = 1.0 - shareSummer
            else:
                shareSummer = 0.0;
                if dayOfYear < m.MID_FALL:
                    dayOfYear += 366
                shareInter = 1 - (dayOfYear - m.MID_FALL) / (m.MID_WINTER - m.MID_FALL + 366)
                shareWinter = 1 - shareInter
            shareSeasons = {'Winter': shareWinter, 'Summer': shareSummer, 'Inter': shareInter}

            # Determine contributions of day values
            lIndex = day.hour * 4  + day.minute // 15
            rIndex = (lIndex + 1) % 96
            share = day.minute / 60

            # PowerConsumption in kWh/a - SLPs are given for a household with 1000kWh/a ->devide by 1000 to receive correcting factor
            if day.weekday() < 5:
                weekdayType = 'Weekday'
            elif day.weekday() < 6:
                weekdayType = 'Saturday'
            else:
                weekdayType = 'Sunday'
    
            # Sum toghether the result
            result = 0
            for season in ['Summer', 'Winter', 'Inter']:
                profile = profiles[(season,weekdayType)]
                result += ((1-share) * profile[lIndex] + share * profile[rIndex]) * shareSeasons[season]
            result *= m.DAY_FACTORS[day.dayofyear-1]
            result *= annual_power
            predictions.loc[day] = result
        return predictions


    def forecast(self, annual_power, timestamps, horizon = pd.Timedelta('1d'), resolution = "15min"):
        '''
        Returns the power for a certain timestamp in Watts.
        
        Parameters
        ----------
        annual_power: 
            The annual_power of the household timeline to predict in KWh/a
        timestamps: [pd.TimeStamp,...] or pd.DatetimeIndex
            The point for which a prognoses shall be done.
        horizont: pd.Timedelta
            If set, the amount of points into the future starting 
            from each timestamp in timestamps.
        resolution: str
            A freq_str representing the frequency with which results
            are returned.

        Returns
        -------
        forecast: pd.DataFrame
            A dataframe with a column for each timestamp in timestamps.
        '''
        m = self.model
        profiles = self.profiles 
        forecasts = pd.DataFrame(columns = timestamps)
        n = len(timestamps)
        for i, day in enumerate(timestamps):
            if self.verbose:
                print("Forecast new Timestamp: {0}/{1}".format(i,n))
            forecast = self._forecast_single(annual_power, day, horizon, resolution)
            forecast.index -= day
            forecasts[day] = forecast
        
        forecasts.index.name = 'horizon'# = forecasts.index.values * pd.Timedelta(resolution)
        #forecasts = forecasts.set_index('horizon')
        return forecasts
