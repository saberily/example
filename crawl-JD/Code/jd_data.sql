/*
Navicat MySQL Data Transfer

Source Server         : studyPython
Source Server Version : 50722
Source Host           : localhost:3306
Source Database       : jd_data

Target Server Type    : MYSQL
Target Server Version : 50722
File Encoding         : 65001

Date: 2018-08-29 09:50:29
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for heat_rankings
-- ----------------------------
DROP TABLE IF EXISTS `heat_rankings`;
CREATE TABLE `heat_rankings` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `book_name` varchar(45) DEFAULT NULL,
  `jd_price` varchar(10) DEFAULT NULL,
  `ding_price` varchar(10) DEFAULT NULL,
  `press` varchar(45) DEFAULT NULL,
  `item_url` varchar(45) DEFAULT NULL,
  `jd_id` varchar(45) DEFAULT NULL,
  `middle_time` varchar(45) DEFAULT NULL,
  `poor_time` varchar(45) DEFAULT NULL,
  `attention_price` varchar(45) DEFAULT NULL,
  `attention` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=101 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of heat_rankings
-- ----------------------------
INSERT INTO `heat_rankings` VALUES ('1', '深入理解Java虚拟机：JVM高级特性与最佳实践（第2版）', '54.50', '79.00', '机械工业出版社', 'http://item.jd.com/11252778.html', '11252778', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('2', '架构即未来：现代企业可扩展的Web架构、流程和组织（原书第2版）', '74.20', '99.00', '机械工业出版社', 'http://item.jd.com/11905648.html', '11905648', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('3', '鸟哥的Linux私房菜 （基础学习篇 第三版） ', '72.20', '88.00', '人民邮电出版社', 'http://item.jd.com/10064429.html', '10064429', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('4', '数学之美（第二版）', '40.20', '49.00', '人民邮电出版社', 'http://item.jd.com/11572052.html', '11572052', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('5', '科技之巅 麻省理工科技评论 50大全球突破性技术深度剖析', '80.40', '98.00', '人民邮电出版社', 'http://item.jd.com/11988131.html', '11988131', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('6', 'C Primer Plus 第6版 中文版 ', '73.00', '89.00', '人民邮电出版社', 'http://item.jd.com/11917487.html', '11917487', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('7', '锋利的jQuery（第2版）', '40.20', '49.00', '人民邮电出版社', 'http://item.jd.com/11019625.html', '11019625', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('8', 'Java从入门到精通（第4版 附光盘）', '59.20', '69.60', '清华大学出版社', 'http://item.jd.com/11985075.html', '11985075', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('9', '机器学习【首届京东文学奖-年度新锐入围作品】 ', '69.50', '88.00', '清华大学出版社', 'http://item.jd.com/11867803.html', '11867803', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('10', '人工智能：一种现代的方法（第3版 影印版）', '151.70', '158.00', '清华大学出版社', 'http://item.jd.com/10779582.html', '10779582', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('11', '视觉机器学习20讲', '39.20', '49.00', '清华大学出版社', 'http://item.jd.com/11706776.html', '11706776', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('12', 'Word Excel PPT 2010办公应用从入门到精通（附DVD光盘1张）', '40.20', '49.00', '人民邮电出版社', 'http://item.jd.com/11266264.html', '11266264', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('13', 'Python基础教程（第2版 修订版） ', '59.20', '79.00', '人民邮电出版社', 'http://item.jd.com/11461683.html', '11461683', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('14', 'Python核心编程（第2版）', '73.00', '89.00', '人民邮电出版社', 'http://item.jd.com/10062788.html', '10062788', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('15', 'Photoshop CS6完全自学教程（中文版 附DVD光盘）', '83.20', '99.00', '人民邮电出版社', 'http://item.jd.com/11020038.html', '11020038', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('16', '大型网站技术架构 核心原理与案例分析', '41.60', '59.00', '电子工业出版社', 'http://item.jd.com/11322972.html', '11322972', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('17', 'Java虚拟机精讲', '55.90', '69.00', '电子工业出版社', 'http://item.jd.com/11631886.html', '11631886', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('18', 'Java核心技术系列：Java多线程编程核心技术', '52.70', '69.00', '机械工业出版社', 'http://item.jd.com/11701869.html', '11701869', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('19', 'Java并发编程的艺术 ', '41.00', '59.00', '机械工业出版社', 'http://item.jd.com/11740734.html', '11740734', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('20', 'Java核心技术系列：Java编程指南', '78.30', '99.00', '机械工业出版社', 'http://item.jd.com/11706272.html', '11706272', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('21', 'Presto技术内幕', '55.90', '69.00', '电子工业出版社', 'http://item.jd.com/11906548.html', '11906548', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('22', 'Excel效率手册：早做完，不加班（套装共3册）', '64.60', '94.00', '清华大学出版社', 'http://item.jd.com/11680843.html', '11680843', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('23', '代码整洁之道 程序员的职业素养', '40.20', '49.00', '人民邮电出版社', 'http://item.jd.com/11977659.html', '11977659', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('24', 'TCP/IP详解卷2：实现', '56.60', '78.00', '机械工业出版社', 'http://item.jd.com/10057318.html', '10057318', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('25', '谁说菜鸟不会数据分析（入门篇）（纪念版）（全彩） ', '47.30', '59.00', '电子工业出版社', 'http://item.jd.com/11944656.html', '11944656', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('26', 'C++ Primer（中文版 第5版） ', '90.00', '128.00', '电子工业出版社', 'http://item.jd.com/11306138.html', '11306138', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('27', '第一行代码 Android 第2版 ', '62.40', '79.00', '人民邮电出版社', 'http://item.jd.com/12012505.html', '12012505', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('28', 'Excel 2013公式与函数辞典646秘技大全（646秘技大全 全新升级版 附 ...', '30.30', '59.90', '中国青年出版社', 'http://item.jd.com/11356071.html', '11356071', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('29', 'PPT炼成记：高效能PPT达人的10堂必修课', '39.70', '49.90', '中国青年出版社', 'http://item.jd.com/11429319.html', '11429319', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('30', 'Excel2013数据透视表实战技巧精粹辞典（339秘技大全 超值双色版 附光盘 ...', '47.60', '59.90', '中国青年出版社', 'http://item.jd.com/11670751.html', '11670751', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('31', 'Office 2013实战技巧精粹辞典（超值双色版 附光盘）', '29.90', '59.00', '中国青年出版社', 'http://item.jd.com/11351487.html', '11351487', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('32', '疯狂Swift讲义（第2版）', '55.90', '69.00', '电子工业出版社', 'http://item.jd.com/11883734.html', '11883734', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('33', 'Cocos2d-x 3.X游戏开发实战（附光盘）', '71.30', '88.00', '电子工业出版社', 'http://item.jd.com/11596624.html', '11596624', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('34', '中文版Photoshop CS5完全自学教程（附光盘）', '91.10', '99.00', '人民邮电出版社', 'http://item.jd.com/10064569.html', '10064569', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('35', '说服力：让你的PPT会说话 工作型PPT该这样做 教你做出专业又出彩的演示PPT ...', '120.50', '147.00', '人民邮电出版社', 'http://item.jd.com/11448625.html', '11448625', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('36', '和秋叶一起学Word', '54.30', '59.00', '人民邮电出版社', 'http://item.jd.com/11799884.html', '11799884', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('37', 'JavaScript权威指南（第6版）', '96.60', '139.00', '机械工业出版社', 'http://item.jd.com/10974436.html', '10974436', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('38', '别怕，Excel VBA其实很简单（全新基础学习版）', '40.20', '49.00', '人民邮电出版社', 'http://item.jd.com/11094550.html', '11094550', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('39', '计算机科学丛书：C程序设计语言（第2版・新版） ', '22.50', '30.00', '机械工业出版社', 'http://item.jd.com/10057446.html', '10057446', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('40', '图解密码技术 第3版', '62.80', '89.00', '人民邮电出版社', 'http://item.jd.com/11942019.html', '11942019', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('41', 'O\'Reilly：Head First设计模式（中文版）', '67.60', '98.00', '中国电力出版社', 'http://item.jd.com/10100236.html', '10100236', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('42', 'C++ Primer Plus（第6版 中文版） ', '81.20', '99.00', '人民邮电出版社', 'http://item.jd.com/11017238.html', '11017238', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('43', 'Excel 2013应用大全', '91.10', '99.00', '人民邮电出版社', 'http://item.jd.com/11731534.html', '11731534', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('44', 'iOS开发指南 从Hello World到App Store上架 第4版', '99.70', '119.00', '人民邮电出版社', 'http://item.jd.com/11976900.html', '11976900', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('45', 'Objective-C基础教程 第2版', '49.60', '59.00', '人民邮电出版社', 'http://item.jd.com/11232703.html', '11232703', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('46', '精通iOS开发（第7版）', '93.20', '118.00', '人民邮电出版社', 'http://item.jd.com/11779940.html', '11779940', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('47', 'iOS9开发快速入门', '57.80', '69.00', '人民邮电出版社', 'http://item.jd.com/11821340.html', '11821340', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('48', 'PHP和MySQL Web开发（原书第4版）', '73.80', '95.00', '机械工业出版社', 'http://item.jd.com/10059047.html', '10059047', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('49', 'HTML5+CSS3从入门到精通（附光盘）', '55.80', '69.80', '清华大学出版社', 'http://item.jd.com/11241686.html', '11241686', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('50', 'Effective Java中文版（第2版） ', '36.10', '52.00', '机械工业出版社', 'http://item.jd.com/10058902.html', '10058902', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('51', 'Axure RP8 实战手册 网站和APP原型制作案例精粹', '64.80', '79.00', '人民邮电出版社', 'http://item.jd.com/12034368.html', '12034368', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('52', 'Axure RP 7.0从入门到精通 Web + APP产品经理原型设计', '81.90', '89.00', '人民邮电出版社', 'http://item.jd.com/11853919.html', '11853919', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('53', 'Axure RP8 网站和APP原型制作 从入门到精通', '81.90', '89.00', '人民邮电出版社', 'http://item.jd.com/11898679.html', '11898679', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('54', 'Axure RP7 网站和APP原型制作从入门到精通 60小时案例版', '99.40', '108.00', '人民邮电出版社', 'http://item.jd.com/11931339.html', '11931339', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('55', '数据结构与算法分析：C语言描述（原书第2版） ', '24.30', '35.00', '机械工业出版社', 'http://item.jd.com/10057441.html', '10057441', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('56', 'HTML CSS JavaScript 网页制作从入门到精通 第3版', '45.10', '49.00', '人民邮电出版社', 'http://item.jd.com/11993156.html', '11993156', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('57', 'Excel 2013数据透视表应用大全', '83.40', '99.00', '北京大学出版社', 'http://item.jd.com/12005808.html', '12005808', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('58', '别怕，Excel VBA其实很简单（第2版）', '40.70', '59.00', '北京大学出版社', 'http://item.jd.com/11974962.html', '11974962', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('59', '别告诉我你懂PPT：全新升级版', '28.90', '42.00', '湖南文艺出版社', 'http://item.jd.com/11807158.html', '11807158', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('60', '设计模式：可复用面向对象软件的基础', '26.20', '35.00', '机械工业出版社', 'http://item.jd.com/10057319.html', '10057319', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('61', 'O\'Reilly：Head First Java（中文版 第2版 涵盖Java5 ...', '45.70', '79.00', '中国电力出版社', 'http://item.jd.com/10100190.html', '10100190', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('62', '编码：隐匿在计算机软硬件背后的语言', '47.80', '59.00', '电子工业出版社', 'http://item.jd.com/11116026.html', '11116026', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('63', 'Spring+MyBatis企业应用实战', '47.00', '58.00', '电子工业出版社', 'http://item.jd.com/12111732.html', '12111732', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('64', '轻量级Java EE企业应用实战：Struts2+Spring4+Hiberna ...', '86.70', '108.00', '电子工业出版社', 'http://item.jd.com/11559101.html', '11559101', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('65', '高性能MySQL（第3版）', '103.70', '128.00', '电子工业出版社', 'http://item.jd.com/11220393.html', '11220393', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('66', '大话设计模式', '35.60', '45.00', '清华大学出版社', 'http://item.jd.com/10079261.html', '10079261', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('67', 'Spring实战（第4版） ', '73.00', '89.00', '人民邮电出版社', 'http://item.jd.com/11899370.html', '11899370', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('68', '软件开发视频大讲堂：Android从入门到精通（附光盘1张）', '55.80', '69.80', '清华大学出版社', 'http://item.jd.com/11078112.html', '11078112', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('69', '机器学习实战 ', '54.50', '69.00', '人民邮电出版社', 'http://item.jd.com/11242112.html', '11242112', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('70', '机器学习系统设计', '38.60', '49.00', '人民邮电出版社', 'http://item.jd.com/11482400.html', '11482400', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('71', 'Docker开发实践', '46.50', '59.00', '人民邮电出版社', 'http://item.jd.com/11717902.html', '11717902', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('72', 'Docker源码分析', '45.90', '59.00', '机械工业出版社', 'http://item.jd.com/11742113.html', '11742113', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('73', '用户体验要素：以用户为中心的产品设计（原书第2版） ', '27.10', '39.00', '机械工业出版社', 'http://item.jd.com/10690653.html', '10690653', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('74', '大话数据结构', '46.60', '59.00', '清华大学出版社', 'http://item.jd.com/10663703.html', '10663703', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('75', '独角兽之路：20款快速爆发且极具潜力的互联网产品深度剖析（全彩）', '64.00', '79.00', '电子工业出版社', 'http://item.jd.com/11933555.html', '11933555', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('76', '编程之美：微软技术面试心得', '32.40', '40.00', '电子工业出版社', 'http://item.jd.com/10066736.html', '10066736', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('77', '数据挖掘 概念与技术（原书第3版） ', '54.90', '79.00', '机械工业出版社', 'http://item.jd.com/11056660.html', '11056660', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('78', 'Hadoop权威指南（第3版 修订版）', '84.20', '99.00', '清华大学出版社', 'http://item.jd.com/11566298.html', '11566298', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('79', '必然 ', '36.60', '58.00', '电子工业出版社', 'http://item.jd.com/11868029.html', '11868029', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('80', '软件开发视频大讲堂：PHP从入门到精通（第3版 附光盘）', '55.80', '69.80', '清华大学出版社', 'http://item.jd.com/11078096.html', '11078096', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('81', '华章专业开发者丛书・Java并发编程实战 ', '47.90', '69.00', '机械工业出版社', 'http://item.jd.com/10922250.html', '10922250', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('82', 'Photoshop CS6从入门到精通（中文版  附光盘）', '91.10', '99.00', '人民邮电出版社', 'http://item.jd.com/11208733.html', '11208733', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('83', 'Excel 2007应用大全', '91.10', '99.00', '人民邮电出版社', 'http://item.jd.com/10931041.html', '10931041', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('84', '软件开发视频大讲堂：C语言从入门到精通（第2版    附光盘）', '42.30', '49.80', '清华大学出版社', 'http://item.jd.com/11078107.html', '11078107', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('85', 'C语言入门经典（第5版）', '58.80', '69.80', '清华大学出版社', 'http://item.jd.com/11362614.html', '11362614', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('86', 'O\'Reilly：深入理解LINUX内核（第3版）（涵盖2.6版）', '65.60', '98.00', '中国电力出版社', 'http://item.jd.com/10100237.html', '10100237', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('87', 'Effective C++：改善程序与设计的55个具体做法（第3版 中文版） ', '51.20', '65.00', '电子工业出版社', 'http://item.jd.com/10393318.html', '10393318', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('88', '黑客与画家：硅谷创业之父Paul Graham文集', '38.60', '49.00', '人民邮电出版社', 'http://item.jd.com/10582495.html', '10582495', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('89', '华为ICT认证系列丛书：HCNP路由交换实验指南', '81.90', '89.00', '人民邮电出版社', 'http://item.jd.com/11585806.html', '11585806', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('90', '华为ICT认证系列丛书：HCNA网络技术学习指南', '48.40', '59.00', '人民邮电出版社', 'http://item.jd.com/11688335.html', '11688335', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('91', 'HTTP权威指南', '81.70', '109.00', '人民邮电出版社', 'http://item.jd.com/11056556.html', '11056556', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('92', '中文版Photoshop CS6基础培训教程（附光盘）', '28.70', '35.00', '人民邮电出版社', 'http://item.jd.com/11047280.html', '11047280', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('93', 'JavaScript语言精粹（修订版）', '23.50', '49.00', '电子工业出版社', 'http://item.jd.com/11090963.html', '11090963', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('94', 'Excel图表之道：如何制作专业有效的商务图表', '47.80', '59.00', '电子工业出版社', 'http://item.jd.com/10067717.html', '10067717', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('95', '简约至上 交互式设计四策略', '26.20', '35.00', '人民邮电出版社', 'http://item.jd.com/10384965.html', '10384965', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('96', '算法(第4版)', '78.20', '99.00', '人民邮电出版社', 'http://item.jd.com/11098789.html', '11098789', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('97', '数据化管理：洞悉零售及电子商务运营 ', '48.10', '59.90', '电子工业出版社', 'http://item.jd.com/11482086.html', '11482086', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('98', 'Auto CAD 2012中文版从入门到精通（附光盘）', '39.70', '49.90', '中国青年出版社', 'http://item.jd.com/10895439.html', '10895439', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('99', '术与道 移动应用UI设计必修课', '54.30', '59.00', '人民邮电出版社', 'http://item.jd.com/11889963.html', '11889963', null, null, null, null);
INSERT INTO `heat_rankings` VALUES ('100', '动静之美 Sketch移动UI与交互动效设计详解', '81.00', '88.00', '人民邮电出版社', 'http://item.jd.com/11993080.html', '11993080', null, null, null, null);

-- ----------------------------
-- Table structure for sales_volume_rankings
-- ----------------------------
DROP TABLE IF EXISTS `sales_volume_rankings`;
CREATE TABLE `sales_volume_rankings` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `book_name` varchar(45) DEFAULT NULL,
  `jd_price` varchar(10) DEFAULT NULL,
  `ding_price` varchar(10) DEFAULT NULL,
  `press` varchar(45) DEFAULT NULL,
  `item_url` varchar(45) DEFAULT NULL,
  `jd_id` varchar(45) DEFAULT NULL,
  `middle_time` varchar(45) DEFAULT NULL,
  `poor_time` varchar(45) DEFAULT NULL,
  `attention_price` varchar(45) DEFAULT NULL,
  `attention` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=101 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of sales_volume_rankings
-- ----------------------------
INSERT INTO `sales_volume_rankings` VALUES ('1', '数学之美（第二版）', '40.20', '49.00', '人民邮电出版社', 'http://item.jd.com/11572052.html', '11572052', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('2', 'Word Excel PPT 2016办公应用从入门到精通（附光盘）', '40.20', '49.00', '人民邮电出版社', 'http://item.jd.com/11988251.html', '11988251', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('3', 'Python编程 从入门到实践 ', '70.30', '89.00', '人民邮电出版社', 'http://item.jd.com/11993134.html', '11993134', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('4', 'C Primer Plus 第6版 中文版 ', '73.00', '89.00', '人民邮电出版社', 'http://item.jd.com/11917487.html', '11917487', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('5', '机器学习【首届京东文学奖-年度新锐入围作品】 ', '69.50', '88.00', '清华大学出版社', 'http://item.jd.com/11867803.html', '11867803', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('6', 'C++ Primer Plus（第6版 中文版） ', '81.20', '99.00', '人民邮电出版社', 'http://item.jd.com/11017238.html', '11017238', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('7', '深度学习 ', '118.50', '168.00', '人民邮电出版社', 'http://item.jd.com/12128543.html', '12128543', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('8', '零基础学Python（全彩版）', '55.80', '79.80', '吉林大学出版社', 'http://item.jd.com/12353915.html', '12353915', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('9', 'Java编程思想（第4版） ', '86.40', '108.00', '机械工业出版社', 'http://item.jd.com/10058164.html', '10058164', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('10', 'ABAQUS分析之美', '99.40', '108.00', '人民邮电出版社', 'http://item.jd.com/12405737.html', '12405737', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('11', 'Spring实战（第4版） ', '73.00', '89.00', '人民邮电出版社', 'http://item.jd.com/11899370.html', '11899370', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('12', '鸟哥的Linux私房菜 （基础学习篇 第三版） ', '72.20', '88.00', '人民邮电出版社', 'http://item.jd.com/10064429.html', '10064429', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('13', 'Python基础教程（第3版）', '78.20', '99.00', '人民邮电出版社', 'http://item.jd.com/12279949.html', '12279949', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('14', 'Python核心编程（第3版） ', '81.20', '99.00', '人民邮电出版社', 'http://item.jd.com/11936238.html', '11936238', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('15', 'Java从入门到精通（第4版 附光盘）', '59.20', '69.60', '清华大学出版社', 'http://item.jd.com/11985075.html', '11985075', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('16', '文明之光（全彩印刷套装1-4册）入选2014中国好书/第六届中华优秀出版物获奖图 ...', '201.70', '246.00', '人民邮电出版社', 'http://item.jd.com/12169220.html', '12169220', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('17', 'Photoshop CC从入门到精通PS教程 全彩高清视频版', '74.90', '99.80', '中国水利水电出版社', 'http://item.jd.com/12216067.html', '12216067', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('18', '代码整洁之道 ', '48.40', '59.00', '人民邮电出版社', 'http://item.jd.com/10064006.html', '10064006', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('19', '深入理解Java虚拟机：JVM高级特性与最佳实践（第2版）', '54.50', '79.00', '机械工业出版社', 'http://item.jd.com/11252778.html', '11252778', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('20', '浪潮之巅 第三版 套装上下册 ', '81.20', '99.00', '人民邮电出版社', 'http://item.jd.com/11922453.html', '11922453', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('21', 'TensorFlow：实战Google深度学习框架（第2版）', '72.10', '89.00', '电子工业出版社', 'http://item.jd.com/12287533.html', '12287533', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('22', '百面机器学习 算法工程师带你去面试', '73.00', '89.00', '人民邮电出版社', 'http://item.jd.com/12401859.html', '12401859', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('23', '华为ICT认证系列丛书：HCNA网络技术学习指南', '48.40', '59.00', '人民邮电出版社', 'http://item.jd.com/11688335.html', '11688335', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('24', '和秋叶一起学Word Excel PPT（套装共3册）', '218.90', '267.00', '人民邮电出版社', 'http://item.jd.com/12191631.html', '12191631', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('25', '区块链核心技术与应用', '78.30', '99.00', '机械工业出版社', 'http://item.jd.com/12419052.html', '12419052', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('26', 'Redis实战', '56.60', '69.00', '人民邮电出版社', 'http://item.jd.com/11791607.html', '11791607', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('27', 'SQL必知必会 第4版 ', '20.50', '29.00', '人民邮电出版社', 'http://item.jd.com/11232698.html', '11232698', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('28', 'C++ Primer（中文版 第5版） ', '90.00', '128.00', '电子工业出版社', 'http://item.jd.com/11306138.html', '11306138', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('29', 'Vue.js实战', '67.20', '79.00', '清华大学出版社', 'http://item.jd.com/12215519.html', '12215519', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('30', '算法导论（原书第3版）/计算机科学丛书', '102.40', '128.00', '机械工业出版社', 'http://item.jd.com/11144230.html', '11144230', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('31', 'HTML5+CSS3+JavaScript从入门到精通（标准版）', '67.40', '89.80', '中国水利水电出版社', 'http://item.jd.com/12123529.html', '12123529', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('32', 'AutoCAD2018从入门到精通CAD教程 实战案例视频版', '44.10', '69.80', '中国水利水电出版社', 'http://item.jd.com/12184539.html', '12184539', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('33', '算法图解 ', '38.70', '49.00', '人民邮电出版社', 'http://item.jd.com/12148832.html', '12148832', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('34', '码农翻身：用故事给技术加点料', '55.40', '69.00', '电子工业出版社', 'http://item.jd.com/12364204.html', '12364204', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('35', '重构 改善既有代码的设计 ', '56.60', '69.00', '人民邮电出版社', 'http://item.jd.com/11728740.html', '11728740', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('36', '零基础学C语言（全彩版 附光盘小白手册）', '48.80', '69.80', '吉林大学出版社', 'http://item.jd.com/12250414.html', '12250414', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('37', '笨办法学Python 3', '48.40', '59.00', '人民邮电出版社', 'http://item.jd.com/12372646.html', '12372646', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('38', '用户体验要素：以用户为中心的产品设计（原书第2版） ', '27.10', '39.00', '机械工业出版社', 'http://item.jd.com/10690653.html', '10690653', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('39', 'Excel高效办公：数据处理与分析（修订版）（附CD光盘1张）', '40.80', '49.80', '人民邮电出版社', 'http://item.jd.com/11042690.html', '11042690', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('40', '利用Python进行数据分析（原书第2版） ', '92.70', '119.00', '机械工业出版社', 'http://item.jd.com/12398725.html', '12398725', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('41', 'JavaScript高级程序设计（第3版） ', '68.30', '99.00', '人民邮电出版社', 'http://item.jd.com/10951037.html', '10951037', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('42', '编程珠玑（第2版 修订版）', '32.00', '39.00', '人民邮电出版社', 'http://item.jd.com/11642529.html', '11642529', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('43', 'Python零基础入门学习-水木书荟', '42.10', '49.50', '清华大学出版社', 'http://item.jd.com/12004711.html', '12004711', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('44', 'C语言从入门到精通（第3版）（附光盘）/软件开发视频大讲堂', '50.80', '59.80', '清华大学出版社', 'http://item.jd.com/12132428.html', '12132428', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('45', 'Hadoop权威指南：大数据的存储与分析(第4版) ', '125.80', '148.00', '清华大学出版社', 'http://item.jd.com/12109713.html', '12109713', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('46', 'Python 3网络爬虫开发实战', '78.20', '99.00', '人民邮电出版社', 'http://item.jd.com/12333540.html', '12333540', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('47', '计算广告：互联网商业变现的市场与技术', '56.60', '69.00', '人民邮电出版社', 'http://item.jd.com/11765659.html', '11765659', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('48', '剑指Offer：名企面试官精讲典型编程题（第2版）', '45.80', '65.00', '电子工业出版社', 'http://item.jd.com/12163054.html', '12163054', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('49', '深入理解计算机系统（原书第3版） ', '111.20', '139.00', '机械工业出版社', 'http://item.jd.com/12006637.html', '12006637', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('50', 'Spring微服务实战', '64.80', '79.00', '人民邮电出版社', 'http://item.jd.com/12357088.html', '12357088', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('51', 'Photoshop CS6从入门到精通PS教程（全彩印 高清视频版）', '79.80', '99.80', '中国水利水电出版社', 'http://item.jd.com/12371603.html', '12371603', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('52', 'Photoshop CS6完全自学教程（中文版 附DVD光盘）', '83.20', '99.00', '人民邮电出版社', 'http://item.jd.com/11020038.html', '11020038', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('53', 'Effective Java中文版（第2版） ', '36.10', '52.00', '机械工业出版社', 'http://item.jd.com/10058902.html', '10058902', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('54', '数据结构与算法分析：Java语言描述（原书第3版） ', '54.50', '69.00', '机械工业出版社', 'http://item.jd.com/11886254.html', '11886254', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('55', 'CSS世界', '56.60', '69.00', '人民邮电出版社', 'http://item.jd.com/12262251.html', '12262251', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('56', '未来地图 : 技术、商业和我们的选择', '79.40', '98.00', '电子工业出版社', 'http://item.jd.com/12403216.html', '12403216', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('57', 'O\'Reilly：Head First Java（中文版 第2版 涵盖Java5 ...', '45.70', '79.00', '中国电力出版社', 'http://item.jd.com/10100190.html', '10100190', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('58', '点石成金：访客至上的Web和移动可用性设计秘笈（原书第3版）', '41.00', '59.00', '机械工业出版社', 'http://item.jd.com/11589225.html', '11589225', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('59', '深度学习入门 基于Python的理论与实现', '46.60', '59.00', '人民邮电出版社', 'http://item.jd.com/12403048.html', '12403048', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('60', 'JavaEE开发的颠覆者：Spring Boot实战 ', '71.40', '89.00', '电子工业出版社', 'http://item.jd.com/11894632.html', '11894632', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('61', 'Python编程快速上手 让繁琐工作自动化', '56.60', '69.00', '人民邮电出版社', 'http://item.jd.com/11943853.html', '11943853', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('62', 'O\'Reilly：Head First设计模式（中文版）', '67.60', '98.00', '中国电力出版社', 'http://item.jd.com/10100236.html', '10100236', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('63', '大话设计模式', '35.60', '45.00', '清华大学出版社', 'http://item.jd.com/10079261.html', '10079261', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('64', '华章专业开发者丛书・Java并发编程实战 ', '47.90', '69.00', '机械工业出版社', 'http://item.jd.com/10922250.html', '10922250', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('65', '高性能MySQL（第3版）', '103.70', '128.00', '电子工业出版社', 'http://item.jd.com/11220393.html', '11220393', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('66', 'Word・Excel・PPT现代商务办公从新手到高手（超值全彩版）（附光盘）', '27.10', '49.90', '中国青年出版社', 'http://item.jd.com/11627789.html', '11627789', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('67', 'PPT炼成记：高效能PPT达人的10堂必修课', '39.70', '49.90', '中国青年出版社', 'http://item.jd.com/11429319.html', '11429319', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('68', '别怕，Excel 函数其实很简单', '40.20', '49.00', '人民邮电出版社', 'http://item.jd.com/11655368.html', '11655368', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('69', '人月神话（40周年中文纪念版）', '53.70', '68.00', '清华大学出版社', 'http://item.jd.com/11671959.html', '11671959', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('70', '第一行代码 Android 第2版 ', '62.40', '79.00', '人民邮电出版社', 'http://item.jd.com/12012505.html', '12012505', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('71', '和秋叶一起学Excel', '81.20', '99.00', '人民邮电出版社', 'http://item.jd.com/12122871.html', '12122871', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('72', '数据结构 Python语言描述', '56.60', '69.00', '人民邮电出版社', 'http://item.jd.com/12273412.html', '12273412', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('73', 'SQL即查即用 （全彩版）', '34.80', '49.80', '吉林大学出版社', 'http://item.jd.com/12359944.html', '12359944', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('74', 'C Primer Plus（第5版 中文版）', '49.20', '60.00', '人民邮电出版社', 'http://item.jd.com/10062260.html', '10062260', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('75', 'UNIX环境高级编程（第3版）', '105.00', '128.00', '人民邮电出版社', 'http://item.jd.com/11469694.html', '11469694', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('76', 'Java 8实战', '55.70', '79.00', '人民邮电出版社', 'http://item.jd.com/11917790.html', '11917790', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('77', '别怕，Excel VBA其实很简单（第2版）', '40.70', '59.00', '北京大学出版社', 'http://item.jd.com/11974962.html', '11974962', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('78', 'Adobe Photoshop CC 2017经典教程 彩色版', '81.20', '99.00', '人民邮电出版社', 'http://item.jd.com/12181615.html', '12181615', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('79', 'Word/Excel/PPT 2016三合一完全自学教程', '65.30', '99.00', '北京大学出版社', 'http://item.jd.com/12273032.html', '12273032', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('80', 'Linux就该这么学', '64.80', '79.00', '人民邮电出版社', 'http://item.jd.com/12269260.html', '12269260', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('81', '启示录：打造用户喜爱的产品', '33.60', '40.00', '华中科技大学出版社', 'http://item.jd.com/12294455.html', '12294455', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('82', '深入浅出Spring Boot 2.x', '81.20', '99.00', '人民邮电出版社', 'http://item.jd.com/12403128.html', '12403128', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('83', 'Spark快速大数据分析 ', '41.60', '59.00', '人民邮电出版社', 'http://item.jd.com/11782888.html', '11782888', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('84', '硅谷之谜：浪潮之巅 续集', '48.40', '59.00', '人民邮电出版社', 'http://item.jd.com/11807307.html', '11807307', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('85', '终极算法：机器学习和人工智能如何重塑世界', '46.90', '68.00', '中信出版社，中信出版集团', 'http://item.jd.com/12079958.html', '12079958', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('86', '零基础学Java（全彩版）（附光盘小白手册）', '48.80', '69.80', '吉林大学出版社', 'http://item.jd.com/12185501.html', '12185501', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('87', 'Go Web编程', '64.80', '79.00', '人民邮电出版社', 'http://item.jd.com/12252845.html', '12252845', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('88', '中文版Photoshop CS6基础培训教程（第2版）', '36.90', '45.00', '人民邮电出版社', 'http://item.jd.com/12339953.html', '12339953', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('89', '未来版图 全球聪明公司的科技创新趋势和商业化路径', '57.20', '69.80', '人民邮电出版社', 'http://item.jd.com/12350389.html', '12350389', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('90', '数据结构与算法分析：C语言描述（原书第2版） ', '24.30', '35.00', '机械工业出版社', 'http://item.jd.com/10057441.html', '10057441', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('91', 'Java核心技术 卷I：基础知识（原书第10版） ', '82.70', '119.00', '机械工业出版社', 'http://item.jd.com/12037418.html', '12037418', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('92', 'Go语言实战', '48.40', '59.00', '人民邮电出版社', 'http://item.jd.com/12136974.html', '12136974', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('93', 'SQL Server 从入门到精通（第2版）（配光盘）（软件开发视频大讲堂）', '63.00', '79.80', '清华大学出版社', 'http://item.jd.com/12165847.html', '12165847', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('94', 'Python神经网络编程', '56.60', '69.00', '人民邮电出版社', 'http://item.jd.com/12335366.html', '12335366', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('95', '精通数据科学 从线性回归到深度学习', '81.20', '99.00', '人民邮电出版社', 'http://item.jd.com/12346637.html', '12346637', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('96', '简约至上 交互式设计四策略', '26.20', '35.00', '人民邮电出版社', 'http://item.jd.com/10384965.html', '10384965', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('97', '算法(第4版)', '78.20', '99.00', '人民邮电出版社', 'http://item.jd.com/11098789.html', '11098789', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('98', '别怕，Excel VBA其实很简单（全新基础学习版）', '40.20', '49.00', '人民邮电出版社', 'http://item.jd.com/11094550.html', '11094550', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('99', '大型网站技术架构 核心原理与案例分析', '41.60', '59.00', '电子工业出版社', 'http://item.jd.com/11322972.html', '11322972', null, null, null, null);
INSERT INTO `sales_volume_rankings` VALUES ('100', '图解HTTP ', '36.70', '49.00', '人民邮电出版社', 'http://item.jd.com/11449491.html', '11449491', null, null, null, null);
