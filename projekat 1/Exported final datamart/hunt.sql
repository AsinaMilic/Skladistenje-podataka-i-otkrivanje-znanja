--------------------------------------------------------
--  File created - Friday-May-12-2023   
--------------------------------------------------------
--------------------------------------------------------
--  DDL for Table HUNT_DIM
--------------------------------------------------------

  CREATE TABLE "S17774"."HUNT_DIM" 
   (	"HUNT_DIM_ID" NUMBER(10,0), 
	"HUNT_ID" NUMBER(10,0), 
	"DATE_FROM" DATE, 
	"DATE_TO" DATE, 
	"HUNT_NAME" VARCHAR2(255 BYTE), 
	"HUNT_DESCRIPTION" VARCHAR2(255 BYTE), 
	"LOCATION" VARCHAR2(255 BYTE), 
	"ANIMALS_HUNTED" NUMBER(10,0), 
	"NUMBER_OF_PEOPLE" NUMBER(10,0), 
	"DURATION" NUMBER(10,0)
   ) SEGMENT CREATION IMMEDIATE 
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 
 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "STUDENTS" ;
REM INSERTING into S17774.HUNT_DIM
SET DEFINE OFF;
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1453,2,to_date('11-MAY-23','DD-MON-RR'),null,'Broadleaf Helleborine','Orchidaceae','Toba',24,30,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1454,3,to_date('11-MAY-23','DD-MON-RR'),null,'Threeflower Rush','Juncaceae','Limbalod',8,13,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1455,4,to_date('11-MAY-23','DD-MON-RR'),null,'California Blackberry','Rosaceae','Ibaiti',8,10,5);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1456,5,to_date('11-MAY-23','DD-MON-RR'),null,'Tapertip Rush','Juncaceae','Jargalant',11,13,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1457,6,to_date('11-MAY-23','DD-MON-RR'),null,'Tiger-pear','Cactaceae','Pushkino',18,5,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1458,7,to_date('11-MAY-23','DD-MON-RR'),null,'Toad Rush','Juncaceae','Ayamaru',10,30,5);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1459,8,to_date('11-MAY-23','DD-MON-RR'),null,'Cuban Raintree','Solanaceae','Curuan',6,8,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1460,12,to_date('11-MAY-23','DD-MON-RR'),null,'Lapland Sedge','Cyperaceae','Itupiranga',11,9,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1461,9,to_date('11-MAY-23','DD-MON-RR'),null,'Woolyleaf Ceanothus','Rhamnaceae','Di�nysos',8,39,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1462,10,to_date('11-MAY-23','DD-MON-RR'),null,'Yellow Prickle','Rutaceae','Tlogosari',23,40,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1463,11,to_date('11-MAY-23','DD-MON-RR'),null,'European Scopolia','Solanaceae','Sh�nm�ri',23,36,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1464,13,to_date('11-MAY-23','DD-MON-RR'),null,'Canary Violet','Violaceae','Zwolle',9,34,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1465,14,to_date('11-MAY-23','DD-MON-RR'),null,'Saxifrage','Saxifragaceae','Labao',13,29,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1466,15,to_date('11-MAY-23','DD-MON-RR'),null,'Gariys Yampah','Apiaceae','Kaniv',7,12,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1467,16,to_date('11-MAY-23','DD-MON-RR'),null,'Wavyleaf Thoroughwort','Asteraceae','Xibing',20,13,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1468,17,to_date('11-MAY-23','DD-MON-RR'),null,'Ghostweed','Urticaceae','Lintaca',23,23,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1469,18,to_date('11-MAY-23','DD-MON-RR'),null,'Cusicks Draba','Brassicaceae','Danderyd',23,30,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1470,19,to_date('11-MAY-23','DD-MON-RR'),null,'Large Hawaii Lovegrass','Poaceae','Kholbon',25,29,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1471,20,to_date('11-MAY-23','DD-MON-RR'),null,'Perennial Rockcress','Brassicaceae','Jianping',21,19,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1472,21,to_date('11-MAY-23','DD-MON-RR'),null,'Tetrapterys','Malpighiaceae','Verkhniye Kigi',25,10,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1473,22,to_date('11-MAY-23','DD-MON-RR'),null,'Jelly Lichen','Collemataceae','Alor Setar',10,44,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1474,25,to_date('11-MAY-23','DD-MON-RR'),null,'Common Serviceberry','Rosaceae','�lvsj�',17,28,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1475,24,to_date('11-MAY-23','DD-MON-RR'),null,'Santessons Map Lichen','Rhizocarpaceae','Kadubadak',15,35,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1476,26,to_date('11-MAY-23','DD-MON-RR'),null,'Plumed Goldenrod','Asteraceae','Quxi',22,43,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1477,27,to_date('11-MAY-23','DD-MON-RR'),null,'Fadyens Silktassel','Garryaceae','Shijing',13,50,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1478,28,to_date('11-MAY-23','DD-MON-RR'),null,'Broadleaf Cattail','Typhaceae','Yangqiao',21,25,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1479,30,to_date('11-MAY-23','DD-MON-RR'),null,'Damiana','Turneraceae','Longshan',12,33,5);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1480,29,to_date('11-MAY-23','DD-MON-RR'),null,'False Sunflower','Asteraceae','K?ty Wroc?awskie',21,35,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1481,32,to_date('11-MAY-23','DD-MON-RR'),null,'Iranian Knapweed','Asteraceae','Xiayang',8,10,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1482,31,to_date('11-MAY-23','DD-MON-RR'),null,'Conejito','Poaceae','Matangshan',23,24,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1483,33,to_date('11-MAY-23','DD-MON-RR'),null,'Datura','Solanaceae','Inari',20,7,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1484,36,to_date('11-MAY-23','DD-MON-RR'),null,'Twisted Airplant','Bromeliaceae','Sauga',6,38,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1485,35,to_date('11-MAY-23','DD-MON-RR'),null,'Baby Blue Eyes','Hydrophyllaceae','Beicheng',7,44,5);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1486,37,to_date('11-MAY-23','DD-MON-RR'),null,'Black River Beardtongue','Scrophulariaceae','Bal�o',24,16,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1487,39,to_date('11-MAY-23','DD-MON-RR'),null,'Soldier Meadows Cinquefoil','Rosaceae','Narowlya',10,26,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1488,40,to_date('11-MAY-23','DD-MON-RR'),null,'Pygmy Pink','Caryophyllaceae','K?chi-shi',25,7,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1489,43,to_date('11-MAY-23','DD-MON-RR'),null,'Great Basin Tumblemustard','Brassicaceae','Anchorage',20,41,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1490,45,to_date('11-MAY-23','DD-MON-RR'),null,'Desert Tobacco','Solanaceae','Mogi Gua�u',7,12,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1491,34,to_date('11-MAY-23','DD-MON-RR'),null,'West Indian False Buttonweed','Rubiaceae','Ipoh',22,34,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1492,38,to_date('11-MAY-23','DD-MON-RR'),null,'Wand Fleabane','Asteraceae','Balucawi',22,29,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1493,41,to_date('11-MAY-23','DD-MON-RR'),null,'Creeping Golden Polypody','Polypodiaceae','Chalchuapa',21,35,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1494,42,to_date('11-MAY-23','DD-MON-RR'),null,'Rocky Mountain Bluebells','Boraginaceae','Mallow',21,25,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1495,44,to_date('11-MAY-23','DD-MON-RR'),null,'Rhachithecium Moss','Rhachitheciaceae','Kerek',9,27,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1496,46,to_date('11-MAY-23','DD-MON-RR'),null,'Hoovers Manzanita','Ericaceae','Juhaynah',18,12,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1497,47,to_date('11-MAY-23','DD-MON-RR'),null,'Waha Milkvetch','Fabaceae','Bariri',14,37,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1498,48,to_date('11-MAY-23','DD-MON-RR'),null,'Cotoneaster','Rosaceae','Pho Duc',8,49,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1499,49,to_date('11-MAY-23','DD-MON-RR'),null,'Lesser Fringed Gentian','Gentianaceae','Dubu',17,14,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1500,50,to_date('11-MAY-23','DD-MON-RR'),null,'Scotch Attorney','Clusiaceae','Sudimara',23,25,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1501,51,to_date('11-MAY-23','DD-MON-RR'),null,'Barberton Daisy','Asteraceae','Odivelas',19,26,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1502,52,to_date('11-MAY-23','DD-MON-RR'),null,'Mule-ears','Asteraceae','Prupuh',17,17,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1503,53,to_date('11-MAY-23','DD-MON-RR'),null,'Wydlers Dancing-lady Orchid','Orchidaceae','Presidente Prudente',14,34,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1504,54,to_date('11-MAY-23','DD-MON-RR'),null,'False Sisal','Agavaceae','Ambohitseheno',14,50,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1505,55,to_date('11-MAY-23','DD-MON-RR'),null,'Summit Lupine','Fabaceae','Kuching',11,10,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1506,56,to_date('11-MAY-23','DD-MON-RR'),null,'Howes Dicranella Moss','Dicranaceae','Taluksangay',13,43,5);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1507,57,to_date('11-MAY-23','DD-MON-RR'),null,'Valley Lessingia','Asteraceae','Guimarei',13,9,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1508,58,to_date('11-MAY-23','DD-MON-RR'),null,'Nantucket Blackberry','Rosaceae','Balkanabat',24,7,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1509,59,to_date('11-MAY-23','DD-MON-RR'),null,'Belllflower African Cornlily','Iridaceae','Hayang',11,33,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1510,60,to_date('11-MAY-23','DD-MON-RR'),null,'Oregon Mock Orange','Hydrangeaceae','Zall-Dardh�',12,35,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1511,61,to_date('11-MAY-23','DD-MON-RR'),null,'American Silvertop','Apiaceae','Turrialba',6,10,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1512,62,to_date('11-MAY-23','DD-MON-RR'),null,'Pochote','Bombacaceae','Si Narong',9,7,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1513,63,to_date('11-MAY-23','DD-MON-RR'),null,'Elmleaf Goldenrod','Asteraceae','Nizhnyaya Maktama',23,28,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1514,64,to_date('11-MAY-23','DD-MON-RR'),null,'Grand Junction Milkvetch','Fabaceae','S�o Paio de Seide',15,37,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1515,65,to_date('11-MAY-23','DD-MON-RR'),null,'Great Basin Woollystar','Polemoniaceae','Akim Oda',5,42,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1517,67,to_date('11-MAY-23','DD-MON-RR'),null,'Longstalk Phacelia','Hydrophyllaceae','Jirny',12,44,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1518,68,to_date('11-MAY-23','DD-MON-RR'),null,'Longtube Cornsalad','Valerianaceae','Goiana',15,45,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1519,69,to_date('11-MAY-23','DD-MON-RR'),null,'Chisos Mountain Pricklypoppy','Papaveraceae','Seseng',20,44,5);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1520,70,to_date('11-MAY-23','DD-MON-RR'),null,'Western Catchfly','Caryophyllaceae','Ho�?ka',15,36,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1521,71,to_date('11-MAY-23','DD-MON-RR'),null,'Plumeless Saw-wort','Asteraceae','Gongfang',8,20,5);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1522,72,to_date('11-MAY-23','DD-MON-RR'),null,'Coralfruit','Cucurbitaceae','Jiuxian',8,29,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1523,74,to_date('11-MAY-23','DD-MON-RR'),null,'Tipularia','Orchidaceae','Sydney',6,20,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1524,76,to_date('11-MAY-23','DD-MON-RR'),null,'Hairyfruit Chervil','Apiaceae','Andarapa',18,11,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1525,78,to_date('11-MAY-23','DD-MON-RR'),null,'Tulare County Rockcress','Brassicaceae','Hikone',14,36,5);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1526,73,to_date('11-MAY-23','DD-MON-RR'),null,'Creeping Waterhyssop','Scrophulariaceae','Mombasa',25,20,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1527,75,to_date('11-MAY-23','DD-MON-RR'),null,'Annual Marsh Elder','Asteraceae','Baton Rouge',18,31,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1528,77,to_date('11-MAY-23','DD-MON-RR'),null,'Coastal Galenia','Aizoaceae','Aguilares',18,14,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1529,79,to_date('11-MAY-23','DD-MON-RR'),null,'Hydrocotyle','Apiaceae','Sandia',23,35,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1530,80,to_date('11-MAY-23','DD-MON-RR'),null,'Stemless Spiderwort','Commelinaceae','Victoria',21,32,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1531,81,to_date('11-MAY-23','DD-MON-RR'),null,'White Blue Eyed Mary','Scrophulariaceae','Y�n Th�nh',15,10,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1532,82,to_date('11-MAY-23','DD-MON-RR'),null,'Dot Lichen','Arthoniaceae','San Crist�bal',19,44,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1533,83,to_date('11-MAY-23','DD-MON-RR'),null,'Vaseys Coastal Pricklypear','Cactaceae','Stepove',7,32,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1534,84,to_date('11-MAY-23','DD-MON-RR'),null,'Blue Potatobush','Solanaceae','Lapai',15,29,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1535,85,to_date('11-MAY-23','DD-MON-RR'),null,'Maritime Quillwort','Isoetaceae','Maebashi-shi',22,39,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1536,86,to_date('11-MAY-23','DD-MON-RR'),null,'Texan Groundcherry','Solanaceae','Maulavi B?z?r',21,6,5);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1537,87,to_date('11-MAY-23','DD-MON-RR'),null,'Mexican Holdback','Fabaceae','Qa?r al Far?firah',5,43,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1538,89,to_date('11-MAY-23','DD-MON-RR'),null,'Wintergreen Barberry','Berberidaceae','Caetit�',23,19,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1539,91,to_date('11-MAY-23','DD-MON-RR'),null,'Thelomma Lichen','Caliciaceae','Juncalito Abajo',16,31,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1540,88,to_date('11-MAY-23','DD-MON-RR'),null,'Setterwort','Ranunculaceae','Honkajoki',24,45,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1541,92,to_date('11-MAY-23','DD-MON-RR'),null,'Helcons Cephalotaxus','Cephalotaxaceae','Ningtang',14,10,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1542,90,to_date('11-MAY-23','DD-MON-RR'),null,'Anomobryum Moss','Bryaceae','Liqiao',22,12,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1543,93,to_date('11-MAY-23','DD-MON-RR'),null,'Winged Lythrum','Lythraceae','Qi�an',20,11,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1544,95,to_date('11-MAY-23','DD-MON-RR'),null,'Millet Crabgrass','Poaceae','Bowang',17,43,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1545,94,to_date('11-MAY-23','DD-MON-RR'),null,'Cochlospermum','Bixaceae','Aroeira',13,18,3);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1546,96,to_date('11-MAY-23','DD-MON-RR'),null,'Gymnoderma Lichen','Cladoniaceae','Galovac',7,30,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1547,97,to_date('11-MAY-23','DD-MON-RR'),null,'Sandwort Orange Lichen','Teloschistaceae','Chubek',11,13,4);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1548,98,to_date('11-MAY-23','DD-MON-RR'),null,'Browse Milkvetch','Fabaceae','Warungbuah',13,8,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1549,99,to_date('11-MAY-23','DD-MON-RR'),null,'Moradia','Verbenaceae','R�o Alejandro',19,50,2);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1550,100,to_date('11-MAY-23','DD-MON-RR'),null,'Alexanders Rock Aster','Asteraceae','Holice',15,45,1);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1452,1,to_date('11-MAY-23','DD-MON-RR'),null,'Geyers Biscuitroot','Apiaceae','Luntas',10,19,5);
Insert into S17774.HUNT_DIM (HUNT_DIM_ID,HUNT_ID,DATE_FROM,DATE_TO,HUNT_NAME,HUNT_DESCRIPTION,LOCATION,ANIMALS_HUNTED,NUMBER_OF_PEOPLE,DURATION) values (1516,66,to_date('11-MAY-23','DD-MON-RR'),null,'Mountain Ladys Slipper','Orchidaceae','�andov',21,50,4);
--------------------------------------------------------
--  DDL for Index SYS_C00235655
--------------------------------------------------------

  CREATE UNIQUE INDEX "S17774"."SYS_C00235655" ON "S17774"."HUNT_DIM" ("HUNT_DIM_ID") 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "STUDENTS" ;
--------------------------------------------------------
--  DDL for Trigger HUNT_DIM_ON_INSERT
--------------------------------------------------------

  CREATE OR REPLACE EDITIONABLE TRIGGER "S17774"."HUNT_DIM_ON_INSERT" 
 BEFORE INSERT ON HUNT_DIM
 FOR EACH ROW
BEGIN
 SELECT HUNT_DIM_S.NEXTVAL
 INTO :NEW.HUNT_DIM_ID
 FROM DUAL;
END;
/
ALTER TRIGGER "S17774"."HUNT_DIM_ON_INSERT" ENABLE;
--------------------------------------------------------
--  Constraints for Table HUNT_DIM
--------------------------------------------------------

  ALTER TABLE "S17774"."HUNT_DIM" ADD PRIMARY KEY ("HUNT_DIM_ID")
  USING INDEX PCTFREE 10 INITRANS 2 MAXTRANS 255 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "STUDENTS"  ENABLE;
