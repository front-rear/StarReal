from pathlib import Path
import cv2,numpy as np,json
B=Path(__file__).resolve().parent
specs={1:dict(source=[433,47,532,252],target=[42,30,137,244],anchor=[90,244],note='Move from right bedside cabinet onto the empty left round table; source lamp appearance retained.',polys=[[[456,49],[506,49],[531,106],[435,106]],[[478,106],[484,106],[483,227],[497,232],[506,245],[459,246],[458,239],[473,227]]]),2:dict(source=[211,15,333,329],target=[507,14,629,328],anchor=[545,326],note='Move standing lamp to the empty floor to the right of the chair.',polys=[[[287,16],[310,21],[328,29],[332,39],[317,56],[301,58],[295,46]],[[238,65],[242,64],[246,309],[249,315],[263,321],[263,327],[216,329],[211,324],[215,318],[236,313]]],lines=[([242,65],[292,25],3)]),3:dict(source=[192,145,238,205],target=[72,226,139,314],anchor=[105,310],note='Move lantern to empty foreground-left blanket, away from the people and book.',polys=[[[205,147],[237,159],[238,188],[232,199],[219,204],[201,200],[194,191],[194,159]]]),4:dict(source=[390,270,451,343],target=[518,244,581,320],anchor=[550,318],note='Move foreground lantern to empty right side of the bed; avoid people.',polys=[[[402,271],[437,271],[445,278],[448,324],[445,337],[438,341],[401,341],[393,335],[392,282],[397,275]]]),5:dict(source=[427,202,476,322],target=[485,202,534,322],anchor=[509,321],note='Move floor lamp to empty floor farther right; source video has camera motion.',polys=[[[428,204],[473,203],[475,224],[429,225]],[[443,224],[450,224],[450,312],[459,317],[458,321],[439,321],[439,317],[444,312]]])}
for i,p in enumerate(sorted(B.glob('case*')),1):
 d=specs[i]; im=cv2.imread(str(p/'source/000000.png'));mask=np.zeros(im.shape[:2],np.uint8)
 for poly in d['polys']:cv2.fillPoly(mask,[np.array(poly,np.int32)],255)
 for a,b,w in d.get('lines',[]):cv2.line(mask,tuple(a),tuple(b),255,w)
 cv2.imwrite(str(p/'source_lamp_mask_first.png'),mask)
 x0,y0,x1,y1=d['source'];srcbox=np.array(d['source']);tarbox=np.array(d['target'])
 # Native first-frame crop and mask retain captured pixels; no generated appearance.
 orig=cv2.imread(str(p/'first_frame_original.jpg'));A=np.array(json.loads((p/'sampling.json').read_text())['resize_crop_matrix']);inv=np.linalg.inv(A)
 coords=np.array([[x0,y0,1],[x1,y1,1]])@inv.T;ox0,oy0=np.floor(coords[0,:2]).astype(int);ox1,oy1=np.ceil(coords[1,:2]).astype(int)
 crop=orig[oy0:oy1,ox0:ox1].copy();mr=cv2.warpPerspective(mask,inv,(orig.shape[1],orig.shape[0]),flags=cv2.INTER_NEAREST)[oy0:oy1,ox0:ox1]
 cv2.imwrite(str(p/'reference_crop_original.png'),crop);cv2.imwrite(str(p/'reference_crop_mask.png'),mr)
 white=np.full_like(crop,255);white[mr>0]=crop[mr>0];rgba=np.dstack([crop,mr]);cv2.imwrite(str(p/'reference_cutout_rgba.png'),rgba)
 scale=460/max(white.shape[:2]);wh=(max(1,round(white.shape[1]*scale)),max(1,round(white.shape[0]*scale)))
 patch=cv2.resize(white,wh,interpolation=cv2.INTER_AREA);ref=np.full((512,512,3),255,np.uint8);rx=(512-wh[0])//2;ry=(512-wh[1])//2;ref[ry:ry+wh[1],rx:rx+wh[0]]=patch
 rm=cv2.resize(mr,wh,interpolation=cv2.INTER_NEAREST);rfull=np.zeros((512,512),np.uint8);rfull[ry:ry+wh[1],rx:rx+wh[0]]=rm
 cv2.imwrite(str(p/'reference.png'),ref);cv2.imwrite(str(p/'reference_mask.png'),rfull)
 over=im.copy();cv2.rectangle(over,tuple(srcbox[:2]),tuple(srcbox[2:]),(0,0,255),2);cv2.rectangle(over,tuple(tarbox[:2]),tuple(tarbox[2:]),(255,0,0),2);cv2.arrowedLine(over,tuple((srcbox[:2]+srcbox[2:])//2),tuple((tarbox[:2]+tarbox[2:])//2),(0,255,255),2)
 cv2.imwrite(str(p/'placement_plan.png'),over)
 (p/'annotation.json').write_text(json.dumps({'source_box_xyxy':d['source'],'target_box_xyxy':d['target'],'target_support_pixel':d['anchor'],'prompt':d['note'],'mask_method':'human/AI visual polygon annotation, approximate silhouette not SAM exact; source RGB preserved','target_geometry':'requested 2D placement; later lifted with estimated source depth; not measured geometry','reference_appearance':'original observed lamp; invisible surfaces unavailable','operation':'move','height_known':False},indent=2))
print('annotations+reference5 complete')
