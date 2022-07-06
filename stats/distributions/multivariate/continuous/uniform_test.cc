// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hybrid_rcc/stats/distributions/multivariate/continuous/uniform.h"

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

class DistributionTest {
 public:
  Eigen::ArrayXd a_, b_;
  Eigen::ArrayXXd x_, pdf_, cdf_;
  DistributionTest() = delete;
  DistributionTest(Eigen::ArrayXd a, Eigen::ArrayXd b, Eigen::ArrayXXd x,
                   Eigen::ArrayXXd pdf, Eigen::ArrayXXd cdf) {
    a_ = a;
    b_ = b;
    x_ = x;
    pdf_ = pdf;
    cdf_ = cdf;
  }
};

class UniformTest : public ::testing::Test {
 protected:
  std::vector<DistributionTest> tests;

  UniformTest() {}

  ~UniformTest() override {}

  void SetUp() override {
    tests = std::vector<DistributionTest>{
        DistributionTest(
            Eigen::ArrayXd{
                {-8.553697015578443, -5.573668226901239, 0.551401209325264}},
            Eigen::ArrayXd{
                {-6.489981836590397, -2.605324667219875, 9.301699518177193}},
            Eigen::ArrayXXd{
                {-8.163689259977692, -4.513864798913458, 6.9084676582563675},
                {-7.66784639095228, -4.309866213357399, 4.948511951253616},
                {-8.002838378825174, -2.984702640617278, 6.657851107876788},
                {-6.781865575454115, -5.090410548584812, 7.716016160043928},
                {-6.526901666999606, -3.2444906853525346, 8.28707757242125}},
            Eigen::ArrayXXd{
                {0.4845629911441343, 0.33688822735443225, 0.11428181813966108},
                {0.4845629911441343, 0.33688822735443225, 0.11428181813966108},
                {0.4845629911441343, 0.33688822735443225, 0.11428181813966108},
                {0.4845629911441343, 0.33688822735443225, 0.11428181813966108},
                {0.4845629911441343, 0.33688822735443225, 0.11428181813966108}},
            Eigen::ArrayXXd{
                {0.18898332462331022, 0.35703529819895413, 0.7264971118184855},
                {0.42925042837575317, 0.4257600200697464, 0.5025098101490062},
                {0.266925708722744, 0.872192027044832, 0.6978561967852172},
                {0.8585639424298633, 0.16280382260343956, 0.8187852228387272},
                {0.9821100165443795, 0.7846724931660973, 0.8840471593146103}}),
        DistributionTest(
            Eigen::ArrayXd{{0.7948694772142462, -0.24722037715596112,
                            -2.4763724804429943, -8.923191142212765,
                            -9.824929735725942}},
            Eigen::ArrayXd{{6.416218175495139, 0.41760400153922017,
                            3.9436508028530537, 4.549181726152083,
                            6.196720032143876}},
            Eigen::ArrayXXd{
                {3.446795801424717, -0.10044606415706411, 2.1415606074604634,
                 -6.231432646189502, 0.4474456582959281},
                {2.24589345903556, 0.1757268447622251, 3.1237009741783437,
                 -7.678244368115832, 2.6837660810540456},
                {5.201125777386141, 0.07700374676733912, 1.1817889121336727,
                 -2.6083254756466605, -0.5576975964824555},
                {2.0204604452831356, 0.1966121222903805, -1.6942645929594824,
                 -6.851830966367002, -5.607289638740533},
                {4.563369174302612, 0.24288894538348244, -1.3691920328369684,
                 3.6974649188600317, -1.1505236013026465}},
            Eigen::ArrayXXd{
                {0.17789325189981856, 1.5041566345124884, 0.15576267497375784,
                 0.07422597413022541, 0.06241554487137915},
                {0.17789325189981856, 1.5041566345124884, 0.15576267497375784,
                 0.07422597413022541, 0.06241554487137915},
                {0.17789325189981856, 1.5041566345124884, 0.15576267497375784,
                 0.07422597413022541, 0.06241554487137915},
                {0.17789325189981856, 1.5041566345124884, 0.15576267497375784,
                 0.07422597413022541, 0.06241554487137915},
                {0.17789325189981856, 1.5041566345124884, 0.15576267497375784,
                 0.07422597413022541, 0.06241554487137915}},
            Eigen::ArrayXXd{
                {0.47175979761253317, 0.2207715566733035, 0.7193016106216682,
                 0.19979839649063721, 0.6411559073412231},
                {0.2581273747108167, 0.6361788698968656, 0.8722824213413527,
                 0.09240738704762652, 0.780737065034664},
                {0.7838432619416413, 0.4876838670682313, 0.5698050039934687,
                 0.46872705560238437, 0.5784193434204389},
                {0.21802436280882143, 0.6675935986544751, 0.12182321667250663,
                 0.1537487268267068, 0.26324630472472066},
                {0.6703906658985306, 0.7372011891341251, 0.1724573881977571,
                 0.936780490295662, 0.5414177853156638}}),
        DistributionTest(
            Eigen::ArrayXd{{-1.109030795008421, -8.285399853432677,
                            -1.6542976020914923, -4.435452007885987,
                            -9.082498129945956, -9.076231448837815}},
            Eigen::ArrayXd{{1.2372017160709214, -1.1411253712582514,
                            6.29900437053225, -1.1494117056749564,
                            -0.17523458626929278, 8.766586106223244}},
            Eigen::ArrayXXd{
                {-0.03847536113725947, -3.140662169707646, 2.7733391298979906,
                 -4.2903225665231, -2.9793516740830075, 5.775600108298912},
                {-0.45495225202554324, -1.154297677049338, 0.5252325355255363,
                 -4.1007826976336, -3.1514519335645303, -3.927497557031},
                {-0.3378664429619931, -3.0676225596633246, 6.106323224262157,
                 -4.060396331930125, -3.893823956284362, -2.6412191455232117},
                {-0.6708732013868202, -3.892735636233448, 2.167973134166294,
                 -2.820596725524781, -0.2302586408915417, -1.2228691245320222},
                {-0.8347328123231355, -5.738664838838712, 2.3897032849522932,
                 -3.7430080871434233, -0.2612183606715792,
                 -6.7250551247983825}},
            Eigen::ArrayXXd{
                {0.4262152174934989, 0.1399722256605741, 0.1257339408766478,
                 0.30431763095758274, 0.11226792550781861, 0.0560449602151737},
                {0.4262152174934989, 0.1399722256605741, 0.1257339408766478,
                 0.30431763095758274, 0.11226792550781861, 0.0560449602151737},
                {0.4262152174934989, 0.1399722256605741, 0.1257339408766478,
                 0.30431763095758274, 0.11226792550781861, 0.0560449602151737},
                {0.4262152174934989, 0.1399722256605741, 0.1257339408766478,
                 0.30431763095758274, 0.11226792550781861, 0.0560449602151737},
                {0.4262152174934989, 0.1399722256605741, 0.1257339408766478,
                 0.30431763095758274, 0.11226792550781861, 0.0560449602151737}},
            Eigen::ArrayXXd{
                {0.4562870170862442, 0.7201203840308195, 0.5567042150832398,
                 0.04416544777775105, 0.6851875916701285, 0.8323703087421891},
                {0.2787782284552781, 0.99815624304134, 0.27404091346201154,
                 0.10184577165021447, 0.6658662525587807, 0.2885605861248294},
                {0.3286819820307014, 0.7303439008101036, 0.9757734401468313,
                 0.11413605478408265, 0.582521685612982, 0.3606500085234202},
                {0.18674943406185873, 0.6148509870609397, 0.48058916276717756,
                 0.49142893386750053, 0.9938225635345311, 0.4401413790210622},
                {0.116909974348237, 0.356472168160432, 0.5084681684366745,
                 0.2107228935313571, 0.990346780020523, 0.13177158353964832}}),
        DistributionTest(Eigen::ArrayXd{{-8.761465502633426}},
                         Eigen::ArrayXd{{-0.5333596939413043}},
                         Eigen::ArrayXXd{{-4.250667597823635},
                                         {-6.956573539603278},
                                         {-3.6771711418602457},
                                         {-2.1933975374849766},
                                         {-8.163726424853456}},
                         Eigen::ArrayXXd{{0.12153465490728207},
                                         {0.12153465490728207},
                                         {0.12153465490728207},
                                         {0.12153465490728207},
                                         {0.12153465490728207}},
                         Eigen::ArrayXXd{{0.5482182667175489},
                                         {0.21935692187179592},
                                         {0.6179179605836088},
                                         {0.7982478735518912},
                                         {0.07264601254258567}}),
        DistributionTest(
            Eigen::ArrayXd{{-8.056182380700399, -5.460953560648704,
                            -1.3420413722518076, 3.3491665420164107,
                            -8.499585525371138, -0.3878443667493525}},
            Eigen::ArrayXd{{5.557591061300529, 3.3796311741164464,
                            2.469724095221224, 8.969738798599373,
                            -4.981976214544204, -0.1493636965815126}},
            Eigen::ArrayXXd{
                {1.1161579818066478, 0.3307371148066771, 2.0036837338305937,
                 4.8266666305642385, -6.878551817906916, -0.20946373545638422},
                {4.777639398620959, 1.9994103803621934, -0.5765752116334212,
                 5.929970484396998, -7.183534479763857, -0.37765623626373124},
                {0.7817752231633985, -5.190567265973503, 1.8643715387171054,
                 4.939837920864858, -7.075301507686259, -0.2655304680946089},
                {-3.6641774045691324, 1.9554701247467623, 2.0613678905274315,
                 8.473649758002406, -7.635447326176369, -0.34083376012753586},
                {4.9002611737845, -4.92258302877082, -0.3789599467877942,
                 8.632197865601247, -7.805503099270719, -0.27994639218870576}},
            Eigen::ArrayXXd{
                {0.07345501996638354, 0.11311468980864474, 0.2623456265956833,
                 0.1779178265751808, 0.2842839871164987, 4.193211966807254},
                {0.07345501996638354, 0.11311468980864474, 0.2623456265956833,
                 0.1779178265751808, 0.2842839871164987, 4.193211966807254},
                {0.07345501996638354, 0.11311468980864474, 0.2623456265956833,
                 0.1779178265751808, 0.2842839871164987, 4.193211966807254},
                {0.07345501996638354, 0.11311468980864474, 0.2623456265956833,
                 0.1779178265751808, 0.2842839871164987, 4.193211966807254},
                {0.07345501996638354, 0.11311468980864474, 0.2623456265956833,
                 0.1779178265751808, 0.2842839871164987, 4.193211966807254}},
            Eigen::ArrayXXd{
                {0.6737544444664207, 0.6551252942217556, 0.8777363493720964,
                 0.26287360451906666, 0.460833925608169, 0.7479877977843071},
                {0.9427086350450582, 0.8438767530470461, 0.20081669954522252,
                 0.4591710282450122, 0.374132238494075, 0.04272099067170096},
                {0.6491923522538664, 0.030584661850694164, 0.8411884042526282,
                 0.2830087945200617, 0.40490113933376326, 0.5128881035459206},
                {0.322614813214178, 0.8389064646629941, 0.8928695356053719,
                 0.911736916109221, 0.24566065268676027, 0.19712543825346995},
                {0.9517158199880096, 0.06089761571548198, 0.2526602000260204,
                 0.9399454508208148, 0.19731651947931986,
                 0.45243907812196865}}),
        DistributionTest(
            Eigen::ArrayXd{{0.24395309991650516, -4.247891923161595,
                            2.3967828807049973, -5.45433989254271,
                            -4.330178084497227}},
            Eigen::ArrayXd{{6.27945141911092, 9.556301074088084,
                            3.929558677887801, 4.883914907657784,
                            -3.2263711798703487}},
            Eigen::ArrayXXd{
                {1.4755153066453035, 0.12845418825702826, 2.891667516791871,
                 -2.978603389059163, -3.5492245320235853},
                {4.252139466242252, -3.0753647771740393, 3.602561594728434,
                 -2.419013836416856, -3.994018588991965},
                {2.9101691071116744, 0.4712505506472704, 2.6959154283352147,
                 4.234812691978292, -3.4957381861793313},
                {4.730598181762646, 7.795339882253333, 3.0949299388970957,
                 -5.083351521177213, -3.586247081030416},
                {2.6384353922821675, 6.594088034183264, 3.377039932593018,
                 3.4309803465486395, -4.157212232045694}},
            Eigen::ArrayXXd{
                {0.16568640187004052, 0.07244175738482055, 0.6524111366045642,
                 0.09672812474892827, 0.9059555578138294},
                {0.16568640187004052, 0.07244175738482055, 0.6524111366045642,
                 0.09672812474892827, 0.9059555578138294},
                {0.16568640187004052, 0.07244175738482055, 0.6524111366045642,
                 0.09672812474892827, 0.9059555578138294},
                {0.16568640187004052, 0.07244175738482055, 0.6524111366045642,
                 0.09672812474892827, 0.9059555578138294},
                {0.16568640187004052, 0.07244175738482055, 0.6524111366045642,
                 0.09672812474892827, 0.9059555578138294}},
            Eigen::ArrayXXd{
                {0.20405311071202162, 0.3170302032353908, 0.3228682479175734,
                 0.23947334935443199, 0.7075092112579499},
                {0.6641019770610651, 0.08493992703674659, 0.7866634613096202,
                 0.2936013974106141, 0.30454556326488513},
                {0.44175573684047365, 0.3418629741520637, 0.19515740539484905,
                 0.9372135599069482, 0.7559654635427043},
                {0.7433760800790007, 0.8724328765770225, 0.4554789157522397,
                 0.03588500946584353, 0.6739684272207768},
                {0.396733155363593, 0.7854120816410634, 0.6395306173869028,
                 0.8594603645210052, 0.15669937534047304}}),
        DistributionTest(
            Eigen::ArrayXd{
                {-3.3202733001952933, 4.469255575454474, -1.2748516206005323,
                 -8.117389757085988, -2.1107459713013976, -9.846954735964966,
                 1.9611853199573992, -4.839175811943233, -9.212814487975592}},
            Eigen::ArrayXd{{4.489004991578513e+00, 5.195309407690156e+00,
                            6.484335628801938e+00, 7.428847976160351e+00,
                            -3.602359152721224e-03, -9.218357536993382e-01,
                            5.113336820575169e+00, -3.199459198142025e+00,
                            5.901750262377825e+00}},
            Eigen::ArrayXXd{
                {-2.404138229546902, 4.807136307635998, 2.043968858730185,
                 -4.346053608505809, -1.7955036370300872, -4.109761717301867,
                 2.081312694127684, -4.4878061803258715, -8.268812787055904},
                {4.2980251877214535, 5.164804166340735, 1.5190339357025415,
                 -4.856310995035522, -1.2484325466672899, -2.2735441580464872,
                 3.00150638117802, -4.148874025964903, -5.79433596306561},
                {2.716065886569193, 4.7816006151349075, 6.389031605959712,
                 -6.555987551568731, -2.0166009567435172, -5.58598232005201,
                 4.318257259814899, -4.580154609912195, 1.3706789938454662},
                {0.22600873387489973, 4.7347046146993765, 4.515255303303146,
                 6.727024496497993, -0.382402167282913, -3.3289963590368474,
                 3.529069810547115, -4.394728888154971, -7.35108049908671},
                {-1.3109274262598745, 4.854196519412834, -0.6148673772401249,
                 -0.13843464809157702, -1.2504244399987021, -7.9862444666968155,
                 3.126439596446404, -4.829922854620703, 3.940660204845855}},
            Eigen::ArrayXXd{
                {0.12805280624374563, 1.377308342166277, 0.12887947768975536,
                 0.06432424469242834, 0.4745761011420999, 0.1120433242388161,
                 0.3172436349598097, 0.6098614794673514, 0.06616134943459867},
                {0.12805280624374563, 1.377308342166277, 0.12887947768975536,
                 0.06432424469242834, 0.4745761011420999, 0.1120433242388161,
                 0.3172436349598097, 0.6098614794673514, 0.06616134943459867},
                {0.12805280624374563, 1.377308342166277, 0.12887947768975536,
                 0.06432424469242834, 0.4745761011420999, 0.1120433242388161,
                 0.3172436349598097, 0.6098614794673514, 0.06616134943459867},
                {0.12805280624374563, 1.377308342166277, 0.12887947768975536,
                 0.06432424469242834, 0.4745761011420999, 0.1120433242388161,
                 0.3172436349598097, 0.6098614794673514, 0.06616134943459867},
                {0.12805280624374563, 1.377308342166277, 0.12887947768975536,
                 0.06432424469242834, 0.4745761011420999, 0.1120433242388161,
                 0.3172436349598097, 0.6098614794673514, 0.06616134943459867}},
            Eigen::ArrayXXd{
                {0.11731366669483866, 0.4653659510908633, 0.42772784992220636,
                 0.2425883492386717, 0.14960647791341308, 0.6428141776107418,
                 0.03810964483995825, 0.21428680337806208,
                 0.062456426401403024},
                {0.9755445001802234, 0.9579848766096466, 0.36007451122129175,
                 0.20976642825141545, 0.40923334302534686, 0.8485500969753997,
                 0.33003523498687637, 0.4209884684756989, 0.22617115222124076},
                {0.7729701723042818, 0.43019542878611816, 0.9877172673143614,
                 0.1004360175309893, 0.04467897395084511, 0.47741351396878684,
                 0.7477660700621632, 0.15796705348406045, 0.7002182104895603},
                {0.45411136619446657, 0.365605176172028, 0.7462259561205422,
                 0.954855734763307, 0.8202306639441981, 0.7302937238012647,
                 0.49740137499179093, 0.27105105848622263, 0.12317483299314656},
                {0.25730237787172194, 0.5301823733552102, 0.0850584245677577,
                 0.5132402608208577, 0.4082880380542343, 0.20848016401410616,
                 0.3696695023258349, 0.005643022242166244,
                 0.8702516354309102}}),
        DistributionTest(
            Eigen::ArrayXd{{-5.533557099474455, -6.2288544087393145,
                            -7.07235747127215, -6.397685664693638,
                            -9.239523826001864}},
            Eigen::ArrayXd{{1.8404052181608233, -2.3008940126062094,
                            -2.1456640400335547, -3.956381774004687,
                            -6.2382412012548905}},
            Eigen::ArrayXXd{
                {-3.48362056789415, -5.585178220140301, -4.656288792884968,
                 -4.961335925522399, -8.756141949547509},
                {-0.17772017579777533, -5.829077941032884, -3.897204827451361,
                 -5.477780106492649, -6.5004106061643725},
                {-0.4721639279987988, -4.694147184015311, -2.555777465491312,
                 -5.511438491540013, -8.000009180502769},
                {-2.588856588120545, -4.282538247866444, -4.517480531942102,
                 -4.5173356415049675, -8.359522218046745},
                {-4.210368879330574, -4.66841101537304, -2.9890520281662294,
                 -6.015636805087756, -8.114331073587053}},
            Eigen::ArrayXXd{
                {0.13561230135505836, 0.2545850515663176, 0.20297589325516344,
                 0.40961717376274437, 0.33319088037712075},
                {0.13561230135505836, 0.2545850515663176, 0.20297589325516344,
                 0.40961717376274437, 0.33319088037712075},
                {0.13561230135505836, 0.2545850515663176, 0.20297589325516344,
                 0.40961717376274437, 0.33319088037712075},
                {0.13561230135505836, 0.2545850515663176, 0.20297589325516344,
                 0.40961717376274437, 0.33319088037712075},
                {0.13561230135505836, 0.2545850515663176, 0.20297589325516344,
                 0.40961717376274437, 0.33319088037712075}},
            Eigen::ArrayXXd{
                {0.2779966106794115, 0.16387033566649073, 0.49040369816146046,
                 0.5883535206941779, 0.16105843297417144},
                {0.7263173709021906, 0.10177711264604183, 0.6444794441010185,
                 0.37680911487892904, 0.9126475451702638},
                {0.6863871760465913, 0.39071351794556075, 0.9167568611317769,
                 0.36302206232240936, 0.41299497597417845},
                {0.399337613146121, 0.495503000180177, 0.5185784289175346,
                 0.7702236621832541, 0.293208510487848},
                {0.1794405996596154, 0.39726556176647276, 0.8288125697480953,
                 0.1564937741110408, 0.37490396377104673}}),
        DistributionTest(
            Eigen::ArrayXd{
                {7.1005862843538425, -3.8484358897690574, 2.0755317282851693,
                 -5.8834016230421105, 1.6067092281419826, -8.26678239755937,
                 -7.840181641072903, -8.806610833763083, 1.9381668208269387}},
            Eigen::ArrayXd{
                {8.533535545514457, 8.041751076519532, 2.6455231599350126,
                 0.41617810376524345, 3.2277211882965506, -0.3418976958168174,
                 6.7115643563331995, 6.249525314751274, 4.866704403139115}},
            Eigen::ArrayXXd{
                {8.089017462950446, -2.6896316519447145, 2.628134315098265,
                 -2.9602258582339873, 2.7727535233785905, -3.524930646920427,
                 2.6611184213533203, -2.114610698139331, 3.2092851935370392},
                {7.906316405249983, 3.592202543933779, 2.1596512216858135,
                 -2.8170052506755354, 2.1694072471543158, -7.789796162502233,
                 -5.623100135571906, -4.034887078987913, 4.546941208673835},
                {8.198041951011138, 6.591388183665778, 2.239325186554775,
                 -2.4446432078771423, 1.6726113895896833, -3.098950672589411,
                 -4.714193638810604, -5.568006688832934, 1.9482899954137665},
                {8.295170089577056, 4.026195528469369, 2.458084193607039,
                 -1.6843940164754585, 2.212255271004107, -6.292338938017597,
                 -0.3333958354187061, 5.532414830839546, 3.5184993780674896},
                {8.0005203894473, -1.951299484250497, 2.5098799835553622,
                 -4.766564519195584, 2.1801197456780397, -2.6657686192734618,
                 -6.576287328102094, 3.6856198955371138, 2.298727253760936}},
            Eigen::ArrayXXd{
                {0.6978614156861714, 0.08410296682762262, 1.7544123375775922,
                 0.15874074833033394, 0.616898594569683, 0.12618480112147454,
                 0.06872027591591094, 0.06641810290076805, 0.34146736106096587},
                {0.6978614156861714, 0.08410296682762262, 1.7544123375775922,
                 0.15874074833033394, 0.616898594569683, 0.12618480112147454,
                 0.06872027591591094, 0.06641810290076805, 0.34146736106096587},
                {0.6978614156861714, 0.08410296682762262, 1.7544123375775922,
                 0.15874074833033394, 0.616898594569683, 0.12618480112147454,
                 0.06872027591591094, 0.06641810290076805, 0.34146736106096587},
                {0.6978614156861714, 0.08410296682762262, 1.7544123375775922,
                 0.15874074833033394, 0.616898594569683, 0.12618480112147454,
                 0.06872027591591094, 0.06641810290076805, 0.34146736106096587},
                {0.6978614156861714, 0.08410296682762262, 1.7544123375775922,
                 0.15874074833033394, 0.616898594569683, 0.12618480112147454,
                 0.06872027591591094, 0.06641810290076805,
                 0.34146736106096587}},
            Eigen::ArrayXXd{
                {0.6897879816037767, 0.09745887437344922, 0.9694927960821875,
                 0.4640271084067377, 0.71933108693746, 0.5983496201018909,
                 0.7216522377657029, 0.444469953619812, 0.43404543632542725},
                {0.5622879628295708, 0.6257797673660436, 0.14758027705286697,
                 0.4867620548268915, 0.3471276170958531, 0.06018841320836568,
                 0.15235845278609175, 0.3169288393586964, 0.8908113058215162},
                {0.7658719651862714, 0.878020177734306, 0.28736126400269707,
                 0.5458710841505202, 0.040654950776190885, 0.6521018184445808,
                 0.21481875802529238, 0.2151019433528245, 0.00345673371172351},
                {0.8336539454688452, 0.6622798649478612, 0.6711547649314118,
                 0.6665536097111548, 0.3735605027888776, 0.24914475526787472,
                 0.5158683918062003, 0.9523708820883312, 0.5396319879196586},
                {0.6280292886047875, 0.1595548001808027, 0.7620259378513279,
                 0.17728755762768048, 0.3537361423794682, 0.706762809691646,
                 0.08685516591590468, 0.829710266038797, 0.12311961953697134}}),
        DistributionTest(
            Eigen::ArrayXd{
                {-4.922068815876591, 2.125193924373856, 2.217349905628805,
                 -4.959082131213856, -5.883631494893744, -7.0422728596655055,
                 -8.06184496750303, 8.48375081065476, -5.012687259803171}},
            Eigen::ArrayXd{
                {-3.8698706194322807, 2.710729903003999, 7.525356321108347,
                 5.6178116935657885, 1.0705078550434366, 9.590395340130582,
                 4.1053118639894, 9.671835498127459, 2.2965866531851553}},
            Eigen::ArrayXXd{
                {-4.532372013469184, 2.191829557844085, 3.204667028305005,
                 -4.37081790936695, -2.7956715006670025, 3.741866869007225,
                 -7.976569735837896, 9.331859528654212, -4.3789475064961545},
                {-4.076631442334421, 2.2485859592930333, 3.4395723820821535,
                 4.04509659839076, -4.033917290732992, 0.710263596766878,
                 -1.494239810664757, 9.046775101366299, -2.5612406905464873},
                {-4.278505442598162, 2.3710717810637494, 4.764542309927315,
                 -0.8645702841120757, -2.3056021629023564, 3.9863310842150312,
                 3.951467985564646, 8.695287073287421, 0.8945740388294787},
                {-4.244726059692146, 2.39981999968188, 5.059028854512436,
                 -3.0860920259114364, -3.6899812724529855, 4.928650360250284,
                 0.06175299107837873, 9.57742834999234, -2.2853121061583823},
                {-4.334889691471931, 2.547030095384542, 3.156592989899454,
                 -2.967961122933171, -4.380158379923305, -5.549444475957067,
                 -3.038893017323855, 9.619973922284201, -1.839232741717555}},
            Eigen::ArrayXXd{
                {0.9503912888078472, 1.707836984397599, 0.1883946479574209,
                 0.09454571602649453, 0.14379924670463115, 0.060122644664568,
                 0.08218846965230901, 0.8416908411867557, 0.13681249490774108},
                {0.9503912888078472, 1.707836984397599, 0.1883946479574209,
                 0.09454571602649453, 0.14379924670463115, 0.060122644664568,
                 0.08218846965230901, 0.8416908411867557, 0.13681249490774108},
                {0.9503912888078472, 1.707836984397599, 0.1883946479574209,
                 0.09454571602649453, 0.14379924670463115, 0.060122644664568,
                 0.08218846965230901, 0.8416908411867557, 0.13681249490774108},
                {0.9503912888078472, 1.707836984397599, 0.1883946479574209,
                 0.09454571602649453, 0.14379924670463115, 0.060122644664568,
                 0.08218846965230901, 0.8416908411867557, 0.13681249490774108},
                {0.9503912888078472, 1.707836984397599, 0.1883946479574209,
                 0.09454571602649453, 0.14379924670463115, 0.060122644664568,
                 0.08218846965230901, 0.8416908411867557, 0.13681249490774108}},
            Eigen::ArrayXXd{
                {0.3703644462842725, 0.1138027993192195, 0.1860052617489165,
                 0.05561786206728434, 0.44404632102384256, 0.6483710009200414,
                 0.007008640789803447, 0.7138453402707796, 0.08670351677214923},
                {0.8034963150470634, 0.21073348081505092, 0.23026017317707576,
                 0.8513065252210004, 0.26598750917717245, 0.46610299461919336,
                 0.5397814171211506, 0.4738923888575716, 0.3353885212730293},
                {0.6116370237596117, 0.41991929729941246, 0.47987741628763436,
                 0.3871185543632025, 0.5145179226274361, 0.6630688360641828,
                 0.9873558070668935, 0.17804813483678703, 0.8081871563378753},
                {0.6437406550147934, 0.4690165682910033, 0.5353571051829454,
                 0.1770831906163567, 0.31544524952042763, 0.7197235630578233,
                 0.6676660842864276, 0.9205383680721092, 0.3731389993195271},
                {0.5580499248040078, 0.7204274142087197, 0.1769483702076111,
                 0.18825196142329312, 0.21619830137341442, 0.08975279045888383,
                 0.41282873392230746, 0.9563485866032173,
                 0.4341682300955362}})};
  }

  void TearDown() override {}
};

TEST_F(UniformTest, VCDF) {
  for (auto test : tests) {
    auto got =
        stats::multivariates::IndependentUniform(test.a_, test.b_).cdf(test.x_);
    auto maxDiff = (got - test.cdf_).abs().maxCoeff();
    EXPECT_THAT(maxDiff, testing::Lt(1e-12));
  }
}
TEST_F(UniformTest, CDF) {
  for (auto test : tests) {
    stats::multivariates::IndependentUniform d(test.a_, test.b_);
    int n = test.x_.rows();
    for (int i = 0; i < n; i++) {
      Eigen::ArrayXd got = d.cdf(Eigen::ArrayXd(test.x_.row(i)));
      EXPECT_THAT((got - (Eigen::ArrayXd)test.cdf_.row(i)).abs().maxCoeff(),
                  testing::Lt(1e-12));
    }
  }
}
TEST_F(UniformTest, VPDF) {
  for (auto test : tests) {
    auto got =
        stats::multivariates::IndependentUniform(test.a_, test.b_).pdf(test.x_);
    auto maxDiff = (got - test.pdf_).abs().maxCoeff();
    EXPECT_THAT(maxDiff, testing::Lt(1e-12));
  }
}
TEST_F(UniformTest, PDF) {
  for (auto test : tests) {
    stats::multivariates::IndependentUniform d(test.a_, test.b_);
    int n = test.x_.rows();
    for (int i = 0; i < n; i++) {
      Eigen::ArrayXd got = d.pdf(Eigen::ArrayXd(test.x_.row(i)));
      EXPECT_THAT((got - (Eigen::ArrayXd)test.pdf_.row(i)).abs().maxCoeff(),
                  testing::Lt(1e-12));
    }
  }
}

TEST_F(UniformTest, PPF) {
  for (auto test : tests) {
    stats::multivariates::IndependentUniform d(test.a_, test.b_);
    int n = test.x_.rows();
    for (int i = 0; i < n; i++) {
      Eigen::ArrayXd got = d.ppf(Eigen::ArrayXd(test.cdf_.row(i)));
      EXPECT_THAT((got - (Eigen::ArrayXd)test.x_.row(i)).abs().maxCoeff(),
                  testing::Lt(1e-12));
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}