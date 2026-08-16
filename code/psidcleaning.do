*PSID cleaning

import excel "/Users/yisroelcahn/Library/Mobile Documents/com~apple~CloudDocs/Documents/Who Moves Up/Data/J359896.xlsx", sheet("Data") firstrow


rename ER30001 ID68
rename ER30002 PN
gen ID =  ID68*1000 + PN

rename V20245  weight1991

rename S217    wealth89

*relation to head
rename ER30465 relation2head85
rename ER30500 relation2head86
rename ER30537 relation2head87
rename ER30572 relation2head88
rename ER30608 relation2head89
rename ER30644 relation2head90
rename ER30691 relation2head91
rename ER30735 relation2head92
rename ER30808 relation2head93
rename ER33103 relation2head94

rename ER34003 relation2head09
rename ER34103 relation2head11
rename ER34203 relation2head13
rename ER34303 relation2head15
rename ER34503 relation2head17
rename ER34703 relation2head19


*age
rename ER30692 age1991
rename ER34504 age2017

*wage
rename V11397 lastyearwage85
rename V12796 lastyearwage86
rename V13898 lastyearwage87
rename V14913 lastyearwage88
rename V16413 lastyearwage89
rename V17829 lastyearwage90
rename V19129 lastyearwage91
rename V20429 lastyearwage92
rename V21739 lastyearwage93
rename ER4122 lastyearwage94

rename ER46811 lastyearwage09
rename ER52219 lastyearwage11
rename ER58020 lastyearwage13
rename ER65200 lastyearwage15
rename ER71277 lastyearwage17
rename ER77299 lastyearwage19

*convert to 2019 dollars (e.g. multiply by cpi2019/cpi1990)
gen gdp83=255.7/99.6 	
gen gdp84=255.7/103.9 	
gen gdp85=255.7/107.6  	
gen gdp86=255.7/109.6 	
gen gdp87=255.7/113.6 	
gen gdp88=255.7/118.3 	
gen gdp89=255.7/124.0  	
gen gdp90=255.7/130.7 	
gen gdp91=255.7/136.2 	
gen gdp92=255.7/140.3 	
gen gdp93=255.7/144.5 	

gen gdp08=255.7/215.3 	
gen gdp10=255.7/218.1 
gen gdp12=255.7/229.6 
gen gdp14=255.7/236.7 
gen gdp16=255.7/240.0 
gen gdp18=255.7/251.1 

replace lastyearwage85=lastyearwage85*gdp84
replace lastyearwage86=lastyearwage86*gdp85
replace lastyearwage87=lastyearwage87*gdp86
replace lastyearwage88=lastyearwage88*gdp87
replace lastyearwage89=lastyearwage89*gdp88
replace lastyearwage90=lastyearwage90*gdp89
replace lastyearwage91=lastyearwage91*gdp90
replace lastyearwage92=lastyearwage92*gdp91
replace lastyearwage93=lastyearwage93*gdp92
replace lastyearwage94=lastyearwage94*gdp93


replace lastyearwage09=lastyearwage09*gdp08
replace lastyearwage11=lastyearwage11*gdp10
replace lastyearwage13=lastyearwage13*gdp12
replace lastyearwage15=lastyearwage15*gdp14
replace lastyearwage17=lastyearwage17*gdp16
replace lastyearwage19=lastyearwage19*gdp18

*gender
rename V19350 gender91
rename ER66018 gender17


*self-employed
rename V19396 employ91
gen selfemployed91=0
replace selfemployed91=1 if employ91==3

*occupation
rename V19401 occ91

*industry
rename V19402 ind91

*race
rename V20114  race91
rename ER70882 race17

*religion
rename V20153  religon91
rename ER70941 religon17

*region
rename V20189  region91
rename ER71530 region17

*state
rename V20190 state91
rename ER66004 state17

*education
rename V20198  edu91
rename ER71538 edu17

*marital status
rename V20216  married91
rename ER71540 married17


*parent variables
gen page1991=0
gen pgender1991=0
gen pselfemployed91=0
gen pocc91=0
gen pind91=0
gen prace91=0
gen preligon91=0
gen pregion91=0
gen pstate91=0
gen pedu91=0
gen pmarried91=0


*============================================================*
* LATER-WAVE SEQUENCE NUMBERS
*============================================================*

rename ER34002 seq09
rename ER34102 seq11
rename ER34202 seq13
rename ER34302 seq15
rename ER34502 seq17
rename ER34702 seq19

*============================================================*
* IDENTIFY CURRENT HEAD / REFERENCE PERSON IN EACH LATER WAVE
*============================================================*

gen head09 = (relation2head09==10 & seq09==1)
gen head11 = (relation2head11==10 & seq11==1)
gen head13 = (relation2head13==10 & seq13==1)
gen head15 = (relation2head15==10 & seq15==1)
gen head17 = (relation2head17==10 & seq17==1)
gen head19 = (relation2head19==10 & seq19==1)

egen n_head_waves = rowtotal(head09 head11 head13 head15 head17 head19)

gen HHWave2 = (n_head_waves >= 1)


*============================================================*
* OFFSPRING LONG-RUN WAGES
*
* Use a wage observation only when the offspring is the
* current household head/reference person in that wave.
*============================================================*

gen childwage09 = lastyearwage09 if head09==1
gen childwage11 = lastyearwage11 if head11==1
gen childwage13 = lastyearwage13 if head13==1
gen childwage15 = lastyearwage15 if head15==1
gen childwage17 = lastyearwage17 if head17==1
gen childwage19 = lastyearwage19 if head19==1

egen wageL0919 = rowmean( childwage09 childwage11 childwage13 childwage15 childwage17 childwage19 )

egen n_child_wagewaves = rownonmiss( childwage09 childwage11 childwage13 childwage15 childwage17 childwage19 )


*============================================================*
* FAMILY INTERVIEW IDs AND SEQUENCE NUMBERS, 1985-1994
*============================================================*

rename ER30463 famid85
rename ER30464 seq85

rename ER30498 famid86
rename ER30499 seq86

rename ER30535 famid87
rename ER30536 seq87

rename ER30570 famid88
rename ER30571 seq88

rename ER30606 famid89
rename ER30607 seq89

rename ER30642 famid90
rename ER30643 seq90

rename ER30689 famid91
rename ER30690 seq91

rename ER30733 famid92
rename ER30734 seq92

rename ER30806 famid93
rename ER30807 seq93

rename ER33101 famid94
rename ER33102 seq94



*============================================================*
* PARENTAL-PERIOD CURRENT-HEAD INDICATORS
*============================================================*

gen head85 = (relation2head85==10 & seq85==1)
gen head86 = (relation2head86==10 & seq86==1)
gen head87 = (relation2head87==10 & seq87==1)
gen head88 = (relation2head88==10 & seq88==1)
gen head89 = (relation2head89==10 & seq89==1)
gen head90 = (relation2head90==10 & seq90==1)
gen head91 = (relation2head91==10 & seq91==1)
gen head92 = (relation2head92==10 & seq92==1)
gen head93 = (relation2head93==10 & seq93==1)
gen head94 = (relation2head94==10 & seq94==1)


*============================================================*
* PARENTAL LONG-RUN WAGES
*
* Use each wage observation only when that individual is
* actually the current household head in that wave.
*============================================================*

gen parentwage85 = lastyearwage85 if head85==1
gen parentwage86 = lastyearwage86 if head86==1
gen parentwage87 = lastyearwage87 if head87==1
gen parentwage88 = lastyearwage88 if head88==1
gen parentwage89 = lastyearwage89 if head89==1
gen parentwage90 = lastyearwage90 if head90==1
gen parentwage91 = lastyearwage91 if head91==1
gen parentwage92 = lastyearwage92 if head92==1
gen parentwage93 = lastyearwage93 if head93==1
gen parentwage94 = lastyearwage94 if head94==1

egen wageL8594 = rowmean( parentwage85 parentwage86 parentwage87 parentwage88 parentwage89 parentwage90 parentwage91 parentwage92 parentwage93 parentwage94 )

egen n_parent_wagewaves = rownonmiss( parentwage85 parentwage86 parentwage87 parentwage88 parentwage89 parentwage90 parentwage91 parentwage92 parentwage93 parentwage94 )





*============================================================*
* PROPER PARENT-CHILD LINKAGE, 1985-1994
*
* A child is linked to the CURRENT HEAD of the SAME FAMILY
* UNIT in a wave in which the child is coded as the head's
* son/daughter (relationship code 30).
*
* Current head requires:
*     relationship to head == 10
*     sequence number == 1
*
* If a child is linked to the same parent in multiple waves,
* those links identify the same parent-person ID.
*
* If different actual parental heads appear across waves,
* choose the parent linked in the greatest number of waves.
* Ties are resolved by choosing the link closest to 1991.
*============================================================*

tempfile masterdata links85 links86 links87 links88 links89 links90 links91 links92 links93 links94 all_links chosen_parent parentchars

* Save full person-level file
save `masterdata', replace


*============================================================*
* 1. 1985 LINK
*============================================================*

preserve

    *--------------------------------------------------------*
    * Create current-head file for 1985
    *--------------------------------------------------------*

    keep if relation2head85==10 & seq85==1 & famid85>0

    keep famid85 ID68 PN

    rename ID68 parent_ID68
    rename PN   parent_PN

    gen parent_ID = parent_ID68*1000 + parent_PN

    isid famid85

    tempfile heads85
    save `heads85', replace

restore


preserve

    * Children actually in the 1985 family
    keep if relation2head85==30 & inrange(seq85,1,20) & famid85>0

    keep ID ID68 PN famid85

    merge m:1 famid85 using `heads85', keep(match) nogen

    gen linkyear = 1985

    keep ID parent_ID parent_ID68 parent_PN linkyear

    save `links85', replace

restore


*============================================================*
* 2. 1986 LINK
*============================================================*

preserve

    keep if relation2head86==10 & seq86==1 & famid86>0

    keep famid86 ID68 PN

    rename ID68 parent_ID68
    rename PN   parent_PN

    gen parent_ID = parent_ID68*1000 + parent_PN

    isid famid86

    tempfile heads86
    save `heads86', replace

restore


preserve

    keep if relation2head86==30 & inrange(seq86,1,20) & famid86>0

    keep ID ID68 PN famid86

    merge m:1 famid86 using `heads86', keep(match) nogen

    gen linkyear = 1986

    keep ID parent_ID parent_ID68 parent_PN linkyear

    save `links86', replace

restore


*============================================================*
* 3. 1987 LINK
*============================================================*

preserve

    keep if relation2head87==10 & seq87==1 & famid87>0

    keep famid87 ID68 PN

    rename ID68 parent_ID68
    rename PN   parent_PN

    gen parent_ID = parent_ID68*1000 + parent_PN

    isid famid87

    tempfile heads87
    save `heads87', replace

restore


preserve

    keep if relation2head87==30 & inrange(seq87,1,20) & famid87>0

    keep ID ID68 PN famid87

    merge m:1 famid87 using `heads87', keep(match) nogen

    gen linkyear = 1987

    keep ID parent_ID parent_ID68 parent_PN linkyear

    save `links87', replace

restore


*============================================================*
* 4. 1988 LINK
*============================================================*

preserve

    keep if relation2head88==10 & seq88==1 & famid88>0

    keep famid88 ID68 PN

    rename ID68 parent_ID68
    rename PN   parent_PN

    gen parent_ID = parent_ID68*1000 + parent_PN

    isid famid88

    tempfile heads88
    save `heads88', replace

restore


preserve

    keep if relation2head88==30 & inrange(seq88,1,20) & famid88>0

    keep ID ID68 PN famid88

    merge m:1 famid88 using `heads88', keep(match) nogen

    gen linkyear = 1988

    keep ID parent_ID parent_ID68 parent_PN linkyear

    save `links88', replace

restore


*============================================================*
* 5. 1989 LINK
*============================================================*

preserve

    keep if relation2head89==10 & seq89==1 & famid89>0

    keep famid89 ID68 PN

    rename ID68 parent_ID68
    rename PN   parent_PN

    gen parent_ID = parent_ID68*1000 + parent_PN

    isid famid89

    tempfile heads89
    save `heads89', replace

restore


preserve

    keep if relation2head89==30 & inrange(seq89,1,20) & famid89>0

    keep ID ID68 PN famid89

    merge m:1 famid89 using `heads89', keep(match) nogen

    gen linkyear = 1989

    keep ID parent_ID parent_ID68 parent_PN linkyear

    save `links89', replace

restore


*============================================================*
* 6. 1990 LINK
*============================================================*

preserve

    keep if relation2head90==10 & seq90==1 & famid90>0

    keep famid90 ID68 PN

    rename ID68 parent_ID68
    rename PN   parent_PN

    gen parent_ID = parent_ID68*1000 + parent_PN

    isid famid90

    tempfile heads90
    save `heads90', replace

restore


preserve

    keep if relation2head90==30 & inrange(seq90,1,20) & famid90>0

    keep ID ID68 PN famid90

    merge m:1 famid90 using `heads90', keep(match) nogen

    gen linkyear = 1990

    keep ID parent_ID parent_ID68 parent_PN linkyear

    save `links90', replace

restore


*============================================================*
* 7. 1991 LINK
*============================================================*

preserve

    keep if relation2head91==10 & seq91==1 & famid91>0

    keep famid91 ID68 PN

    rename ID68 parent_ID68
    rename PN   parent_PN

    gen parent_ID = parent_ID68*1000 + parent_PN

    isid famid91

    tempfile heads91
    save `heads91', replace

restore


preserve

    keep if relation2head91==30 & inrange(seq91,1,20) & famid91>0

    keep ID ID68 PN famid91

    merge m:1 famid91 using `heads91', keep(match) nogen

    gen linkyear = 1991

    keep ID parent_ID parent_ID68 parent_PN linkyear

    save `links91', replace

restore


*============================================================*
* 8. 1992 LINK
*============================================================*

preserve

    keep if relation2head92==10 & seq92==1 & famid92>0

    keep famid92 ID68 PN

    rename ID68 parent_ID68
    rename PN   parent_PN

    gen parent_ID = parent_ID68*1000 + parent_PN

    isid famid92

    tempfile heads92
    save `heads92', replace

restore


preserve

    keep if relation2head92==30 & inrange(seq92,1,20) & famid92>0

    keep ID ID68 PN famid92

    merge m:1 famid92 using `heads92', keep(match) nogen

    gen linkyear = 1992

    keep ID parent_ID parent_ID68 parent_PN linkyear

    save `links92', replace

restore


*============================================================*
* 9. 1993 LINK
*============================================================*

preserve

    keep if relation2head93==10 & seq93==1 & famid93>0

    keep famid93 ID68 PN

    rename ID68 parent_ID68
    rename PN   parent_PN

    gen parent_ID = parent_ID68*1000 + parent_PN

    isid famid93

    tempfile heads93
    save `heads93', replace

restore


preserve

    keep if relation2head93==30 & inrange(seq93,1,20) & famid93>0

    keep ID ID68 PN famid93

    merge m:1 famid93 using `heads93', keep(match) nogen

    gen linkyear = 1993

    keep ID parent_ID parent_ID68 parent_PN linkyear

    save `links93', replace

restore


*============================================================*
* 10. 1994 LINK
*============================================================*

preserve

    keep if relation2head94==10 & seq94==1 & famid94>0

    keep famid94 ID68 PN

    rename ID68 parent_ID68
    rename PN   parent_PN

    gen parent_ID = parent_ID68*1000 + parent_PN

    isid famid94

    tempfile heads94
    save `heads94', replace

restore


preserve

    keep if relation2head94==30 & inrange(seq94,1,20) & famid94>0

    keep ID ID68 PN famid94

    merge m:1 famid94 using `heads94', keep(match) nogen

    gen linkyear = 1994

    keep ID parent_ID parent_ID68 parent_PN linkyear

    save `links94', replace

restore


*============================================================*
* 11. COMBINE ALL CHILD-PARENT LINKS
*============================================================*

use `links85', clear

append using `links86'
append using `links87'
append using `links88'
append using `links89'
append using `links90'
append using `links91'
append using `links92'
append using `links93'
append using `links94'


* Remove exact duplicate child-parent-year records
duplicates drop ID parent_ID linkyear, force


*============================================================*
* 12. CHOOSE ONE PARENTAL HEAD PER CHILD
*
* First preference:
*   parent linked to child in greatest number of waves.
*
* Tie breaker:
*   parent-child link closest to 1991.
*
* This avoids arbitrary ID68 overwriting.
*============================================================*

bysort ID parent_ID: gen n_linkwaves = _N


* Distance of each observed link from 1991
gen dist1991 = abs(linkyear - 1991)


* Minimum distance to 1991 for each child-parent pair
bysort ID parent_ID: egen mindist1991 = min(dist1991)


* Keep one row per child-parent candidate
bysort ID parent_ID (dist1991): keep if _n==1


* Sort best candidate first:
*  - most linked waves
*  - closest to 1991
*  - parent_ID only as final deterministic tie-break
gsort ID -n_linkwaves mindist1991 parent_ID


* Diagnostic: how many distinct parental heads per child?
bysort ID: gen n_parent_candidates = _N

tab n_parent_candidates


* Keep preferred parental head
by ID: keep if _n==1


rename linkyear parent_linkyear

keep ID parent_ID parent_ID68 parent_PN parent_linkyear n_linkwaves n_parent_candidates

isid ID

save `chosen_parent', replace


*============================================================*
* 13. CREATE PARENT CHARACTERISTICS FILE
*============================================================*

use `masterdata', clear

keep ID age1991 gender91 selfemployed91 occ91 ind91 race91 religon91 region91 state91 edu91 married91 wageL8594 n_parent_wagewaves wealth89

rename ID parent_ID

rename age1991        page1991
rename gender91       pgender1991
rename selfemployed91 pselfemployed91
rename occ91          pocc91
rename ind91          pind91
rename race91         prace91
rename religon91      preligon91
rename region91       pregion91
rename state91        pstate91
rename edu91          pedu91
rename married91      pmarried91

rename wageL8594 parent_wageL8594
rename n_parent_wagewaves parent_n_wagewaves
rename wealth89  parent_wealth89

isid parent_ID

save `parentchars', replace


*============================================================*
* 14. RETURN TO FULL PERSON FILE AND MERGE PARENT ID
*============================================================*

use `masterdata', clear

drop page1991 pgender1991 pselfemployed91 pocc91 pind91 prace91 preligon91 pregion91 pstate91 pedu91 pmarried91

merge 1:1 ID using `chosen_parent', keep(master match) gen(parent_link_merge)

tab parent_link_merge


*============================================================*
* 15. MERGE PARENTAL CHARACTERISTICS
*============================================================*

merge m:1 parent_ID using `parentchars', keep(master match) gen(parent_char_merge)

tab parent_char_merge


*============================================================*
* 16. CREATE CHILD VARIABLE FROM PROPER LINKAGE
*
* child = 1 if the individual was observed as the
* son/daughter of an actual current household head in at
* least one wave from 1985-1994 and was successfully linked
* to that parental head.
*============================================================*

gen child = (parent_link_merge==3)

tab child


*============================================================*
* 17. REPLACE PARENTAL INCOME AND WEALTH
*============================================================*

replace wageL8594 = parent_wageL8594 if child==1 & parent_char_merge==3
replace wealth89   = parent_wealth89  if child==1 & parent_char_merge==3

drop parent_wageL8594 parent_wealth89


*============================================================*
* 18. DIAGNOSTICS
*============================================================*

display "================================================"
display "PROPER PARENT-CHILD LINKAGE DIAGNOSTICS"
display "================================================"

count if child==1
display "Children linked to an actual parental head: " r(N)

count if child==1 & n_parent_candidates==1
display "Children with one parental-head candidate: " r(N)

count if child==1 & n_parent_candidates>1
display "Children with multiple parental-head candidates: " r(N)

tab n_parent_candidates if child==1

tab parent_linkyear if child==1

count if child==1 & parent_char_merge==3
display "Linked children with parental characteristics: " r(N)

count if child==1 & parent_char_merge!=3
display "Linked children missing parental characteristics: " r(N)

count if child==1 & wageL8594!=. & wageL8594>0
display "Linked children with valid parental long-run income: " r(N)

count if child==1 & wealth89!=.
display "Linked children with observed parental wealth: " r(N)




drop parent_link_merge parent_char_merge


*============================================================*
* FINAL LINKAGE / WAGE DIAGNOSTICS
*============================================================*

count if child==1
display "Properly linked children: " r(N)

count if child==1 & HHWave2==1
display "Linked children observed as later head/reference person: " r(N)

tab n_head_waves if child==1 & HHWave2==1

tab n_child_wagewaves if child==1 & HHWave2==1

summarize n_parent_wagewaves n_child_wagewaves if child==1 & HHWave2==1

count if child==1 & HHWave2==1 & inrange(age2017,25,64) & wageL0919>0 & wageL0919<. & wageL8594>0 & wageL8594<.

display "Potential final analysis sample before other covariate requirements: " r(N)

summarize parent_n_wagewaves n_child_wagewaves if child==1 & HHWave2==1

tab parent_n_wagewaves if child==1 & HHWave2==1


*============================================================*
* ECONOMICALLY MEANINGFUL ANALYSIS SAMPLE
*============================================================*

* Fix missing age
replace page1991 = . if page1991==0 | page1991==999
replace age2017  = . if age2017==0 | age2017==999

* Must be a properly linked child
keep if child==1

* Must be observed as an adult current head/reference person
* in at least one outcome-period wave
keep if HHWave2==1

* Adult age restriction
keep if inrange(age2017,25,64)

* Positive long-run offspring wages
keep if wageL0919>0 & wageL0919<.

* Positive parental long-run wages
keep if wageL8594>0 & wageL8594<.

keep if parent_n_wagewaves >= 2
keep if n_child_wagewaves >= 2


*age groups
gen agegr_4_2017 = 0
replace agegr_4_2017 = 1 if age2017<=34  
replace agegr_4_2017 = 2 if age2017>34 & age2017<=44  
replace agegr_4_2017 = 3 if age2017>44 & age2017<=54
replace agegr_4_2017 = 4 if age2017>54

gen pagegr_4_1991 = .
replace pagegr_4_1991 = 1 if inrange(page1991,25,34)
replace pagegr_4_1991 = 2 if inrange(page1991,35,44)
replace pagegr_4_1991 = 3 if inrange(page1991,45,54)
replace pagegr_4_1991 = 4 if inrange(page1991,55,64)

gen agegr_2_2017 = 0
replace agegr_2_2017 = 1 if age2017<=44  
replace agegr_2_2017 = 2 if age2017>44  

gen pagegr_2_1991 = .
replace pagegr_2_1991 = 1 if inrange(page1991,25,44)
replace pagegr_2_1991 = 2 if inrange(page1991,45,64)


*percentile variables 
xtile wage8594_p = wageL8594 [aw=weight1991], nq(100)
xtile wage0919_p = wageL0919 [aw=weight1991], nq(100)

*quantile variables
xtile wage8594_q = wageL8594 [aw=weight1991], nq(5)
xtile wage0919_q = wageL0919 [aw=weight1991], nq(5)

*education groups
gen edugr_5_2017 = 0
replace edugr_5_2017 = 1 if edu17<=12  
replace edugr_5_2017 = 2 if edu17==12 
replace edugr_5_2017 = 3 if edu17>12 & edu17<16
replace edugr_5_2017 = 4 if edu17==16
replace edugr_5_2017 = 5 if edu17>16 & edu17<=17
replace edugr_5_2017 = . if edugr_5_2017==0

gen pedugr_5_1991 = 0
replace pedugr_5_1991 = 1 if pedu91<=12  
replace pedugr_5_1991 = 2 if pedu91==12 
replace pedugr_5_1991 = 3 if pedu91>12 & pedu91<16
replace pedugr_5_1991 = 4 if pedu91==16
replace pedugr_5_1991 = 5 if pedu91>16 & pedu91<=17
replace pedugr_5_1991 = . if pedugr_5_1991==0


****************************
*********Table 1************
****************************

sum page1991 [aw=weight1991]
sum age2017 [aw=weight1991]

tab pagegr_4_1991 [aw=weight1991]
tab agegr_4_2017 [aw=weight1991]

tab prace91 [aw=weight1991]
tab race17 [aw=weight1991]

tab pgender1991 [aw=weight1991]
tab gender17 [aw=weight1991]

sum wageL8594 [aw=weight1991]
sum wageL0919 [aw=weight1991]

tabstat wageL8594 [aw=weight1991], by(wage8594_q) stat(mean sd min max) columns(statistics)
tabstat wageL0919  [aw=weight1991], by(wage0919_q) stat(mean sd min max) columns(statistics)

tab pedugr_5_1991 [aw=weight1991]
tab edugr_5_2017 [aw=weight1991]

tab pregion91 [aw=weight1991]
tab region17 [aw=weight1991]


****************************
*********Table 2************
****************************
*wealth

*covert to 2019 dollars
replace wealth89=wealth89*gdp88

xtile wealth89_q = wealth89 [aw=weight1991], nq(5)

sum wealth89 [aw=weight1991]

tabstat wealth89 [aw=weight1991], by(wealth89_q) stat(mean sd min max) columns(statistics)

corr wealth89 wageL8594 [aw=weight1991]
corr wealth89 wageL0919 [aw=weight1991]
corr wageL8594 wageL0919 [aw=weight1991]

****************************
*********Table 3************
****************************

gen age2017squared= age2017^2
gen page1991squared= page1991^2

gen wage2017_log = log(wageL0919)
gen pwage1991_log = log(wageL8594)

*overall
regress wage2017_log pwage1991_log age2017 age2017squared page1991 page1991squared [aw=weight1991], cluster(ID68)

*4 groups
forvalues i=1/4{
	regress wage2017_log pwage1991_log age2017 age2017squared page1991 page1991squared [aw=weight1991] if pagegr_4_1991==`i'  
	est store sig_rs_age4_`i'
}
/* clustered std.errors handled in suest command */

suest sig_rs_age4_1 sig_rs_age4_2 sig_rs_age4_3 sig_rs_age4_4, vce(cluster ID68)
test [sig_rs_age4_1_mean]pwage1991_log = [sig_rs_age4_2_mean]pwage1991_log, cons
test [sig_rs_age4_1_mean]pwage1991_log= [sig_rs_age4_3_mean]pwage1991_log, cons
test [sig_rs_age4_1_mean]pwage1991_log = [sig_rs_age4_4_mean]pwage1991_log, cons	

*2 groups
forvalues i=1/2{
	regress wage2017_log pwage1991_log age2017 age2017squared page1991 page1991squared [aw=weight1991] if pagegr_2_1991==`i'  
	est store sig_rs_age2_`i'
}
/* clustered std.errors handled in suest command */

suest sig_rs_age2_1 sig_rs_age2_2, vce(cluster ID68)
test [sig_rs_age2_1_mean]pwage1991_log = [sig_rs_age2_2_mean]pwage1991_log 

****************************
*********Table 4************
****************************

*overall
regress wage0919_p wage8594_p age2017 age2017squared page1991 page1991squared [aw=weight1991], cluster(ID68)

*4 groups

forvalues i=1/4{
	regress wage0919_p wage8594_p age2017 age2017squared page1991 page1991squared [aw=weight1991] if pagegr_4_1991==`i'  
	est store sig_rs_age4_`i'
}
/* clustered std.errors handled in suest command */

suest sig_rs_age4_1 sig_rs_age4_2 sig_rs_age4_3 sig_rs_age4_4, vce(cluster ID68)
test [sig_rs_age4_1_mean]wage8594_p = [sig_rs_age4_2_mean]wage8594_p, cons
test [sig_rs_age4_1_mean]wage8594_p = [sig_rs_age4_3_mean]wage8594_p, cons
test [sig_rs_age4_1_mean]wage8594_p = [sig_rs_age4_4_mean]wage8594_p, cons

*2 groups
forvalues i=1/2{
	regress wage0919_p wage8594_p age2017 age2017squared page1991 page1991squared [aw=weight1991] if pagegr_2_1991==`i'  
	est store sig_rs_age2_`i'
}
/* clustered std.errors handled in suest command */

suest sig_rs_age2_1 sig_rs_age2_2, vce(cluster ID68)
test [sig_rs_age2_1_mean]wage8594_p = [sig_rs_age2_2_mean]wage8594_p 


****************************
*********Table 5************
****************************

tab wage8594_q wage0919_q [aw=weight1991]


* Quintile Boundaries
	* Parents
tabstat wageL8594 [aw=weight1991], by(wage8594_q) stat(min max median mean n) columns(statistics)
	* Children
tabstat wageL0919 [aw=weight1991], by(wage0919_q) stat(min max median mean n) columns(statistics)


*============================================================*
* APPENDIX TABLE
* LIFECYCLE ROBUSTNESS: IGE AND RANK-RANK
*============================================================*

preserve

*------------------------------------------------------------*
* CONSTRUCT MEAN CHILD AGE AT OBSERVED EARNINGS
*------------------------------------------------------------*

capture drop cage09 cage11 cage13 cage15 cage17 cage19
capture drop cage_w09 cage_w11 cage_w13 cage_w15 cage_w17 cage_w19
capture drop mean_child_earn_age

gen cage09 = age2017 - 8
gen cage11 = age2017 - 6
gen cage13 = age2017 - 4
gen cage15 = age2017 - 2
gen cage17 = age2017
gen cage19 = age2017 + 2

gen cage_w09 = cage09 if childwage09>0 & childwage09<.
gen cage_w11 = cage11 if childwage11>0 & childwage11<.
gen cage_w13 = cage13 if childwage13>0 & childwage13<.
gen cage_w15 = cage15 if childwage15>0 & childwage15<.
gen cage_w17 = cage17 if childwage17>0 & childwage17<.
gen cage_w19 = cage19 if childwage19>0 & childwage19<.

egen mean_child_earn_age = rowmean(cage_w09 cage_w11 cage_w13 cage_w15 cage_w17 cage_w19)


*------------------------------------------------------------*
* CREATE RESULTS FILE
*------------------------------------------------------------*

tempfile lifecycle_table
tempname results

postfile `results' str60 specification ige ige_se rank_slope rank_se N using `lifecycle_table', replace


*============================================================*
* SPECIFICATION 1: BASELINE
*============================================================*

gen sample1 = wage2017_log<. & pwage1991_log<. & weight1991>0 & weight1991<.

quietly reg wage2017_log pwage1991_log [aw=weight1991] if sample1==1, cluster(ID68)

local ige = _b[pwage1991_log]
local igese = _se[pwage1991_log]
local n = e(N)

xtile prank1 = wageL8594 [aw=weight1991] if sample1==1, nq(100)
xtile crank1 = wageL0919 [aw=weight1991] if sample1==1, nq(100)

quietly reg crank1 prank1 [aw=weight1991] if sample1==1, cluster(ID68)

local rank = _b[prank1]
local rankse = _se[prank1]

post `results' ("Baseline sample") (`ige') (`igese') (`rank') (`rankse') (`n')


*============================================================*
* SPECIFICATION 2: >=4 PARENT / >=4 CHILD WAVES
*============================================================*

gen sample2 = sample1==1 & parent_n_wagewaves>=4 & n_child_wagewaves>=4

quietly reg wage2017_log pwage1991_log [aw=weight1991] if sample2==1, cluster(ID68)

local ige = _b[pwage1991_log]
local igese = _se[pwage1991_log]
local n = e(N)

xtile prank2 = wageL8594 [aw=weight1991] if sample2==1, nq(100)
xtile crank2 = wageL0919 [aw=weight1991] if sample2==1, nq(100)

quietly reg crank2 prank2 [aw=weight1991] if sample2==1, cluster(ID68)

local rank = _b[prank2]
local rankse = _se[prank2]

post `results' ("At least 4 parent / 4 child wage waves") (`ige') (`igese') (`rank') (`rankse') (`n')


*============================================================*
* SPECIFICATION 3: >=6 PARENT / >=4 CHILD WAVES
*============================================================*

gen sample3 = sample1==1 & parent_n_wagewaves>=6 & n_child_wagewaves>=4

quietly reg wage2017_log pwage1991_log [aw=weight1991] if sample3==1, cluster(ID68)

local ige = _b[pwage1991_log]
local igese = _se[pwage1991_log]
local n = e(N)

xtile prank3 = wageL8594 [aw=weight1991] if sample3==1, nq(100)
xtile crank3 = wageL0919 [aw=weight1991] if sample3==1, nq(100)

quietly reg crank3 prank3 [aw=weight1991] if sample3==1, cluster(ID68)

local rank = _b[prank3]
local rankse = _se[prank3]

post `results' ("At least 6 parent / 4 child wage waves") (`ige') (`igese') (`rank') (`rankse') (`n')


*============================================================*
* SPECIFICATION 4: >=6 PARENT / >=6 CHILD WAVES
*============================================================*

gen sample4 = sample1==1 & parent_n_wagewaves>=6 & n_child_wagewaves>=6

quietly reg wage2017_log pwage1991_log [aw=weight1991] if sample4==1, cluster(ID68)

local ige = _b[pwage1991_log]
local igese = _se[pwage1991_log]
local n = e(N)

xtile prank4 = wageL8594 [aw=weight1991] if sample4==1, nq(100)
xtile crank4 = wageL0919 [aw=weight1991] if sample4==1, nq(100)

quietly reg crank4 prank4 [aw=weight1991] if sample4==1, cluster(ID68)

local rank = _b[prank4]
local rankse = _se[prank4]

post `results' ("At least 6 parent / 6 child wage waves") (`ige') (`igese') (`rank') (`rankse') (`n')


*============================================================*
* SPECIFICATION 5: PARENT AGE 30-54
*============================================================*

gen sample5 = sample1==1 & inrange(page1991,30,54)

quietly reg wage2017_log pwage1991_log [aw=weight1991] if sample5==1, cluster(ID68)

local ige = _b[pwage1991_log]
local igese = _se[pwage1991_log]
local n = e(N)

xtile prank5 = wageL8594 [aw=weight1991] if sample5==1, nq(100)
xtile crank5 = wageL0919 [aw=weight1991] if sample5==1, nq(100)

quietly reg crank5 prank5 [aw=weight1991] if sample5==1, cluster(ID68)

local rank = _b[prank5]
local rankse = _se[prank5]

post `results' ("Parent age 30-54") (`ige') (`igese') (`rank') (`rankse') (`n')


*============================================================*
* SPECIFICATION 6: PARENT AGE 35-54
*============================================================*

gen sample6 = sample1==1 & inrange(page1991,35,54)

quietly reg wage2017_log pwage1991_log [aw=weight1991] if sample6==1, cluster(ID68)

local ige = _b[pwage1991_log]
local igese = _se[pwage1991_log]
local n = e(N)

xtile prank6 = wageL8594 [aw=weight1991] if sample6==1, nq(100)
xtile crank6 = wageL0919 [aw=weight1991] if sample6==1, nq(100)

quietly reg crank6 prank6 [aw=weight1991] if sample6==1, cluster(ID68)

local rank = _b[prank6]
local rankse = _se[prank6]

post `results' ("Parent age 35-54") (`ige') (`igese') (`rank') (`rankse') (`n')


*============================================================*
* SPECIFICATION 7: PARENT AGE 40-54
*============================================================*

gen sample7 = sample1==1 & inrange(page1991,40,54)

quietly reg wage2017_log pwage1991_log [aw=weight1991] if sample7==1, cluster(ID68)

local ige = _b[pwage1991_log]
local igese = _se[pwage1991_log]
local n = e(N)

xtile prank7 = wageL8594 [aw=weight1991] if sample7==1, nq(100)
xtile crank7 = wageL0919 [aw=weight1991] if sample7==1, nq(100)

quietly reg crank7 prank7 [aw=weight1991] if sample7==1, cluster(ID68)

local rank = _b[prank7]
local rankse = _se[prank7]

post `results' ("Parent age 40-54") (`ige') (`igese') (`rank') (`rankse') (`n')


*============================================================*
* SPECIFICATION 8: JOINT LIFECYCLE RESTRICTION
*============================================================*

gen sample8 = sample1==1 & inrange(page1991,40,54) & mean_child_earn_age>=35 & mean_child_earn_age<. & parent_n_wagewaves>=6 & n_child_wagewaves>=4

quietly reg wage2017_log pwage1991_log [aw=weight1991] if sample8==1, cluster(ID68)

local ige = _b[pwage1991_log]
local igese = _se[pwage1991_log]
local n = e(N)

xtile prank8 = wageL8594 [aw=weight1991] if sample8==1, nq(100)
xtile crank8 = wageL0919 [aw=weight1991] if sample8==1, nq(100)

quietly reg crank8 prank8 [aw=weight1991] if sample8==1, cluster(ID68)

local rank = _b[prank8]
local rankse = _se[prank8]

post `results' ("Parent 40-54; child mean earnings age >=35") (`ige') (`igese') (`rank') (`rankse') (`n')


*============================================================*
* DISPLAY RESULTS
*============================================================*

postclose `results'

use `lifecycle_table', clear

format ige ige_se rank_slope rank_se %6.3f
format N %8.0f

list specification ige ige_se rank_slope rank_se N, noobs clean





postclose `results'

use `lifecycle_table', clear

format ige ige_se rank_slope rank_se %6.3f
format N %8.0f

list specification ige ige_se rank_slope rank_se N, noobs clean

restore

****************************
******Create Features*******
****************************

*---------------------------*
* Parent variables
*---------------------------*

gen pfemale91 = 0
replace pfemale91 = 1 if pgender1991==2

* Occupations
gen poccTechnical = 0
replace poccTechnical = 1 if pocc91>=1 & pocc91<=195

gen poccManager = 0
replace poccManager = 1 if pocc91>=201 & pocc91<=245

gen poccSales = 0
replace poccSales = 1 if pocc91>=260 & pocc91<=285

gen poccClerical = 0
replace poccClerical = 1 if pocc91>=301 & pocc91<=395

gen poccCraftsman = 0
replace poccCraftsman = 1 if pocc91>=401 & pocc91<=600

gen poccOperatives = 0
replace poccOperatives = 1 if pocc91>=601 & pocc91<=695

gen poccTransport = 0
replace poccTransport = 1 if pocc91>=701 & pocc91<=715

gen poccLaborers = 0
replace poccLaborers = 1 if pocc91>=740 & pocc91<=785

gen poccFarmers = 0
replace poccFarmers = 1 if pocc91>=801 & pocc91<=802

gen poccFarmLaborers = 0
replace poccFarmLaborers = 1 if pocc91>=821 & pocc91<=824

gen poccService = 0
replace poccService = 1 if pocc91>=901 & pocc91<=965

gen poccPrivate = 0
replace poccPrivate = 1 if pocc91>=980 & pocc91<=984

gen poccOther = 0
replace poccOther = 1 if pocc91==0 | pocc91==999


* Industry
gen pindAgriculture = 0
replace pindAgriculture = 1 if pind91>=17 & pind91<=28

gen pindMining = 0
replace pindMining = 1 if pind91>=47 & pind91<=57

gen pindConstruction = 0
replace pindConstruction = 1 if pind91>=67 & pind91<=77

gen pindManufacturing = 0
replace pindManufacturing = 1 if pind91>=107 & pind91<=398

gen pindTransportation = 0
replace pindTransportation = 1 if pind91>=407 & pind91<=479

gen pindRetailTrade = 0
replace pindRetailTrade = 1 if pind91>=507 & pind91<=698

gen pindFinance = 0
replace pindFinance = 1 if pind91>=707 & pind91<=718

gen pindBusiness = 0
replace pindBusiness = 1 if pind91>=727 & pind91<=759

gen pindPersonal = 0
replace pindPersonal = 1 if pind91>=769 & pind91<=798

gen pindEntertainment = 0
replace pindEntertainment = 1 if pind91>=807 & pind91<=809

gen pindProfessional = 0
replace pindProfessional = 1 if pind91>=828 & pind91<=897

gen pindPublic = 0
replace pindPublic = 1 if pind91>=907 & pind91<=937

gen pindOther = 0
replace pindOther = 1 if pind91==0 | pind91==999


* Race
gen pWhite = 0
replace pWhite = 1 if prace91==1

gen pBlack = 0
replace pBlack = 1 if prace91==2


* Region
gen pNortheast = 0
replace pNortheast = 1 if pregion91==1

gen pNorthCentral = 0
replace pNorthCentral = 1 if pregion91==2

gen pSouth = 0
replace pSouth = 1 if pregion91==3

gen pWest = 0
replace pWest = 1 if pregion91==4

gen pOther = 0
replace pOther = 1 if pregion91==5 | pregion91==6 | pregion91==9


* Education
gen psomeHS = 0
replace psomeHS = 1 if pedu91<12

gen pHS = 0
replace pHS = 1 if pedu91==12

gen psomeCollege = 0
replace psomeCollege = 1 if pedu91>12 & pedu91<16

gen pCollege = 0
replace pCollege = 1 if pedu91==16

gen ppostgrad = 0
replace ppostgrad = 1 if pedu91>16 & pedu91<=17


* Married
gen pmarried = 0
replace pmarried = 1 if pmarried91==1



****************************
******Child Variables*******
****************************

gen cfemale = 0
replace cfemale = 1 if gender91==2


* Occupations
gen coccTechnical = 0
replace coccTechnical = 1 if occ91>=1 & occ91<=195

gen coccManager = 0
replace coccManager = 1 if occ91>=201 & occ91<=245

gen coccSales = 0
replace coccSales = 1 if occ91>=260 & occ91<=285

gen coccClerical = 0
replace coccClerical = 1 if occ91>=301 & occ91<=395

gen coccCraftsman = 0
replace coccCraftsman = 1 if occ91>=401 & occ91<=600

gen coccOperatives = 0
replace coccOperatives = 1 if occ91>=601 & occ91<=695

gen coccTransport = 0
replace coccTransport = 1 if occ91>=701 & occ91<=715

gen coccLaborers = 0
replace coccLaborers = 1 if occ91>=740 & occ91<=785

gen coccFarmers = 0
replace coccFarmers = 1 if occ91>=801 & occ91<=802

gen coccFarmLaborers = 0
replace coccFarmLaborers = 1 if occ91>=821 & occ91<=824

gen coccService = 0
replace coccService = 1 if occ91>=901 & occ91<=965

gen coccPrivate = 0
replace coccPrivate = 1 if occ91>=980 & occ91<=984

gen coccOther = 0
replace coccOther = 1 if occ91==0 | occ91==999


* Industry
gen cindAgriculture = 0
replace cindAgriculture = 1 if ind91>=17 & ind91<=28

gen cindMining = 0
replace cindMining = 1 if ind91>=47 & ind91<=57

gen cindConstruction = 0
replace cindConstruction = 1 if ind91>=67 & ind91<=77

gen cindManufacturing = 0
replace cindManufacturing = 1 if ind91>=107 & ind91<=398

gen cindTransportation = 0
replace cindTransportation = 1 if ind91>=407 & ind91<=479

gen cindRetailTrade = 0
replace cindRetailTrade = 1 if ind91>=507 & ind91<=698

gen cindFinance = 0
replace cindFinance = 1 if ind91>=707 & ind91<=718

gen cindBusiness = 0
replace cindBusiness = 1 if ind91>=727 & ind91<=759

gen cindPersonal = 0
replace cindPersonal = 1 if ind91>=769 & ind91<=798

gen cindEntertainment = 0
replace cindEntertainment = 1 if ind91>=807 & ind91<=809

gen cindProfessional = 0
replace cindProfessional = 1 if ind91>=828 & ind91<=897

gen cindPublic = 0
replace cindPublic = 1 if ind91>=907 & ind91<=937

gen cindOther = 0
replace cindOther = 1 if ind91==0 | ind91==999


* Race
gen cWhite = 0
replace cWhite = 1 if race91==1

gen cBlack = 0
replace cBlack = 1 if race91==2


* Region
gen cNortheast = 0
replace cNortheast = 1 if region91==1

gen cNorthCentral = 0
replace cNorthCentral = 1 if region91==2

gen cSouth = 0
replace cSouth = 1 if region91==3

gen cWest = 0
replace cWest = 1 if region91==4

gen cOther = 0
replace cOther = 1 if region91==5 | region91==6 | region91==9


* Education
gen csomeHS = 0
replace csomeHS = 1 if edu91<12

gen cHS = 0
replace cHS = 1 if edu91==12

gen csomeCollege = 0
replace csomeCollege = 1 if edu91>12 & edu91<16

gen cCollege = 0
replace cCollege = 1 if edu91==16

gen cpostgrad = 0
replace cpostgrad = 1 if edu91>16 & edu91<=17


* Married
gen cmarried = 0
replace cmarried = 1 if married91==1



****************************
****Dependent Variables*****
****************************

* Rank-change outcomes
gen DV1 = (wage0919_p-wage8594_p>=10) if !missing(wage0919_p,wage8594_p)
gen DV2 = (wage0919_p-wage8594_p>=20) if !missing(wage0919_p,wage8594_p)
gen DV3 = (wage0919_p-wage8594_p>=30) if !missing(wage0919_p,wage8594_p)
gen DV4 = (wage0919_p-wage8594_p>=40) if !missing(wage0919_p,wage8594_p)

gen DV5 = (wage0919_p-wage8594_p<=-10) if !missing(wage0919_p,wage8594_p)
gen DV6 = (wage0919_p-wage8594_p<=-20) if !missing(wage0919_p,wage8594_p)
gen DV7 = (wage0919_p-wage8594_p<=-30) if !missing(wage0919_p,wage8594_p)
gen DV8 = (wage0919_p-wage8594_p<=-40) if !missing(wage0919_p,wage8594_p)



****************************
**********Wealth************
****************************

* DO NOT drop observations with missing wealth.
* Percentile is missing only when wealth is missing.

xtile wealth_p = wealth89 [aw=weight1991], nq(100)



****************************
*****Common variable list***
****************************

local Xvars wage8594_p wage0919_p wageL8594 wageL0919 wealth89 wealth_p page1991 age2017 pselfemployed91 selfemployed91 poccTechnical poccManager poccSales poccClerical poccCraftsman poccOperatives poccTransport poccLaborers poccFarmers poccFarmLaborers poccService poccPrivate pindAgriculture pindMining pindConstruction pindManufacturing pindTransportation pindRetailTrade pindFinance pindBusiness pindPersonal pindEntertainment pindProfessional pindPublic pBlack pNortheast pNorthCentral pSouth pWest psomeHS pHS psomeCollege pCollege pmarried pfemale91 cfemale coccTechnical coccManager coccSales coccClerical coccCraftsman coccOperatives coccTransport coccLaborers coccFarmers coccFarmLaborers coccService coccPrivate cindAgriculture cindMining cindConstruction cindManufacturing cindTransportation cindRetailTrade cindFinance cindBusiness cindPersonal cindEntertainment cindProfessional cindPublic cBlack cNortheast cNorthCentral cSouth cWest csomeHS cHS csomeCollege cCollege cmarried weight1991 ID68



*****************
*****Data 1******
*****************

preserve

keep DV1 DV2 DV3 DV4 DV5 DV6 DV7 DV8 `Xvars'

save "/Users/yisroelcahn/Library/Mobile Documents/com~apple~CloudDocs/Documents/Who Moves Up/Data/psidcleaned_data1.dta", replace

restore



******************
*****Data 2*******
******************

preserve

* Absolute parent-child income difference outcomes
gen I1 = (wageL0919>=wageL8594+5000) if !missing(wageL0919,wageL8594)
gen I2 = (wageL0919>=wageL8594+10000) if !missing(wageL0919,wageL8594)
gen I3 = (wageL0919>=wageL8594+15000) if !missing(wageL0919,wageL8594)
gen I4 = (wageL0919>=wageL8594+20000) if !missing(wageL0919,wageL8594)

gen I5 = (wageL0919<=wageL8594-5000) if !missing(wageL0919,wageL8594)
gen I6 = (wageL0919<=wageL8594-10000) if !missing(wageL0919,wageL8594)
gen I7 = (wageL0919<=wageL8594-15000) if !missing(wageL0919,wageL8594)
gen I8 = (wageL0919<=wageL8594-20000) if !missing(wageL0919,wageL8594)

keep I1 I2 I3 I4 I5 I6 I7 I8 `Xvars'

save "/Users/yisroelcahn/Library/Mobile Documents/com~apple~CloudDocs/Documents/Who Moves Up/Data/psidcleaned_data2.dta", replace

restore



******************
*****Data 3*******
******************

preserve

* Weighted parental-generation median
summarize wageL8594 [aw=weight1991], detail
scalar median_wage = r(p50)

display "Weighted parental median income = " median_wage


* Mobility relative to parental-generation median
gen A1 = (wageL0919>=median_wage+5000) if !missing(wageL0919)
gen A2 = (wageL0919>=median_wage+10000) if !missing(wageL0919)
gen A3 = (wageL0919>=median_wage+15000) if !missing(wageL0919)
gen A4 = (wageL0919>=median_wage+20000) if !missing(wageL0919)

gen A5 = (wageL0919<=median_wage-5000) if !missing(wageL0919)
gen A6 = (wageL0919<=median_wage-10000) if !missing(wageL0919)
gen A7 = (wageL0919<=median_wage-15000) if !missing(wageL0919)
gen A8 = (wageL0919<=median_wage-20000) if !missing(wageL0919)

keep A1 A2 A3 A4 A5 A6 A7 A8 `Xvars'

save "/Users/yisroelcahn/Library/Mobile Documents/com~apple~CloudDocs/Documents/Who Moves Up/Data/psidcleaned_data3.dta", replace

restore



****************************
*****Quick verification*****
****************************

count
summarize wageL8594 wageL0919 wealth89
tab DV1
tab DV5

preserve
use "/Users/yisroelcahn/Library/Mobile Documents/com~apple~CloudDocs/Documents/Who Moves Up/Data/psidcleaned_data1.dta", clear
count
describe
restore
