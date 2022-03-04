clear all;

xform2 = [0.032448869, -0.99917442, 0.024444092, -35.249733;
          -0.9993943, -0.032128867, 0.013372214, 2.2048476;
          -0.012575814, -0.024863198, -0.99961174, 124.63991;
          0, 0, 0, 1];


camMat = [607.32593, 0, 547.92737;
          0, 609.42407, 377.49368;
          0, 0, 1];

object_points = [-32, -22.4, 0;
                 -32, -9.5999994, 0;
                 -32, 3.2000008, 0;
                 -32, 16.000002, 0;
                 -25.6, -16, 0;
                 -25.6, -3.1999989, 0;
                 -25.6, 9.6000004, 0;
                 -25.6, 22.4, 0;
                 -19.200001, -22.4, 0;
                 -19.200001, -9.5999994, 0;
                 -19.200001, 3.2000008, 0;
                 -19.200001, 16.000002, 0;
                 -12.799999, -16, 0;
                 -12.799999, -3.1999989, 0;
                 -12.799999, 9.6000004, 0;
                 -12.799999, 22.4, 0;
                 -6.3999996, -22.4, 0;
                 -6.3999996, -9.5999994, 0;
                 -6.3999996, 3.2000008, 0;
                 -6.3999996, 16.000002, 0;
                 0, -16, 0;
                 0, -3.1999989, 0;
                 0, 9.6000004, 0;
                 0, 22.4, 0;
                 6.4000015, -22.4, 0;
                 6.4000015, -9.5999994, 0;
                 6.4000015, 3.2000008, 0;
                 6.4000015, 16.000002, 0;
                 12.799999, -16, 0;
                 12.799999, -3.1999989, 0;
                 12.799999, 9.6000004, 0;
                 12.799999, 22.4, 0;
                 19.200001, -22.4, 0;
                 19.200001, -9.5999994, 0;
                 19.200001, 3.2000008, 0;
                 19.200001, 16.000002, 0;
                 25.600002, -16, 0;
                 25.600002, -3.1999989, 0;
                 25.600002, 9.6000004, 0;
                 25.600002, 22.4, 0;
                 32, -22.4, 0;
                 32, -9.5999994, 0;
                 32, 3.2000008, 0;
                 32, 16.000002, 0];

image_points = [777.12384, 464.3674;
                729.7843, 468.43033;
                682.23187, 472.40851;
                634.52594, 476.34006;
                751.69977, 442.59265;
                704.15057, 446.60922;
                656.59888, 450.53973;
                609.00916, 454.58777;
                773.73303, 416.91269;
                726.02661, 420.75854;
                678.56122, 424.6846;
                630.91748, 428.5683;
                747.86975, 394.94714;
                700.55231, 398.82895;
                652.92285, 402.70389;
                605.18842, 406.58652;
                770.01892, 369.26047;
                722.55261, 373.06732;
                674.98206, 376.89127;
                627.22662, 380.6933;
                744.37537, 347.15411;
                696.8031, 351.00488;
                649.18207, 354.80896;
                601.39362, 358.58148;
                766.24249, 321.47879;
                718.88849, 325.16351;
                671.15063, 329.03409;
                623.45593, 332.73465;
                740.81195, 299.39932;
                692.96075, 303.16339;
                645.35614, 306.85672;
                597.54596, 310.55664;
                762.46729, 273.6962;
                714.9292, 277.33905;
                667.3465, 280.94949;
                619.4541, 284.81219;
                736.76605, 251.64458;
                689.23773, 255.25259;
                641.45435, 258.93036;
                593.61255, 262.61172;
                758.56335, 225.93694;
                710.9812, 229.52394;
                663.34625, 233.12984;
                615.64734, 236.57167];

% % Converting from model coordinates to camera coordinates
% %step1 = xform2*object_points(5,:)';

% step1 = xform2*object_point;
% % Converting from camera coordinates to image plane positions
% step2 = camMat*step1;

% x = step2(1,1)/step2(3,1);
% y = step2(2,1)/step2(3,1);
% step2(3,1)

% ip1 = [x y];

% normVal = sqrt(xform2(1,4)*xform2(1,4) + xform2(2,4)*xform2(2,4) + xform2(3,4)*xform2(3,4))
% ip3 = [x*normVal, y*normVal, normVal];

% rStep1 = inv(camMat)*ip3';
% rOP = inv([xform2; 0 0 0 1])*[rStep1;1];

%% Using ray plane intersection:
MVorign = [0 0 0 1];
MVunitx = [1 0 0 1];
MVunity = [0 1 0 1];
MVunitz = [0 0 1 1];

CCMVorign = xform2*MVorign';
CCMVunitx = xform2*MVunitx';
CCMVunity = xform2*MVunity';
CCMVunitz = xform2*MVunitz';

CCMVaxisx = CCMVunitx - CCMVorign;
CCMVaxisy = CCMVunity - CCMVorign;
CCMVaxisz = CCMVunitz - CCMVorign;

% Normal to the plane
CCMVzaxis = normalize(CCMVaxisz);

% Distance from camera origin to plane along plane's normal
MVCorigin = [0 0 0 1];
CCMVorigin = xform2*MVCorigin';

distance = CCMVorigin'*CCMVzaxis;

[maxRows cols] = size(image_points);

diffs = [];
rmse = [];
for iter = 1:maxRows
    % Sample image position
    ip = [image_points(iter,1), image_points(iter,2), 1];

    % Convert image position to camera coordinates
    CCip = inv(camMat)*ip';

    % Direction of ray
    CCorigin = [0 0 0];
    CCray = normalize(CCip);

    % Ray plane intersection
    % ray is CCorigin + t*CCray
    % plane is (x,y,z)*CCMVzaxis - distance = 0
    % put (x,y,z) = CCorigin + t*CCray
    t = -(CCorigin*CCMVzaxis - distance)/(CCray'*CCMVzaxis);

    % Intersecting point on plane
    op = CCorigin' + t*CCray;

    % Convert op to MVC
    rop = inv([xform2; 0 0 0 1])*[op;1];


    % diff = rOP - object_point

    % ip4 = [x, y, 1];

    % rStep1 = inv(camMat)*ip4';
    % rOP = inv([xform2; 0 0 0 1])*[rStep1;1];

    diff = rop - [object_points(iter,:) 1]';
    diffs = [diffs; diff'];
    rmse = [rmse; sum(diff.*diff)];
end


