%%  Erstellung von Vertices, Faces, FaceVertexCData Matrizen
%   zur Erzeungung einiger geomtrischer Standartformen:
%
%   Mögliche Objekte sind:
%   - 'Kegel'
%   - 'Kugel'
%   - 'Zylinder'
%   - 'Koordinatensystem'
%   - 'Pfeil' (Länge 1 entlang z-Achse)
%   - 'Pfeil_mit_Kugel' (Länge 1 entlang z-Achse mit Kugel am Nicht-Pfeil-Ende)
%   - 'Elliptic_cone' (Requires the size of the X axis and Y axis of
%   the elliptic base: func_create_VFC_data('Elliptic_cone',12,x,y))
%   - 'Pyramide_updown' (Requires the size of the X axis and Y axis of
%   the rectangular base: func_create_VFC_data('Pyramide_updown',[],x,y))
%
%   Objekte haben in der Regel das in jeder Raumrichtung als Länge 1
%   Skalierung kann nachträglich durch Veränderung der Vertices-Matrix
%   erfolgen Standardfarbe ist grau [0.5,0.5,0.5] (Aunahme: Koordinatensystem)
%
%   Parameter:
%   - Objekt
%       Zur Erzeungung des gewünschten Objekts muss nur die entsprechende
%       Bezeichnung als string im Parameter "Objekt" übergeben werden.
%   - Teilung
%       Bei rotationssymmethrischen Objekten enscheidet der Parameter "Teilung"
%       in wieviele diskrete Punkte ein entsprechender Kreisquerschnitt
%       geteilt wird.
%       Wird kein Wert für "Teilung" gesetzt ist der Default 36.
%       Werte kleiner 12 werden auf 12 korrigert.
%       Dezimalzahlen werden aufgerundet.
%
%
% patch('Faces', F,...
%             'Vertices' ,V,...
%             'FaceVertexCData', C,...
%             'FaceC', 'flat',...
%             'EdgeColor','none');

function [V,F,C] = func_create_VFC_data(Objekt,Teilung, varargin)

if ~(1 == exist('Objekt','var'))
    disp('kein geometrisches Objekt gewählt');
    return;
end

if ~(1 == exist('Teilung','var'))
    Teilung = 36;
elseif Teilung < 12
    Teilung = 12;
else
    Teilung = ceil(Teilung);
end


func = str2func(Objekt);
[V,F,C] = func(Teilung,varargin{:});

end

function [vertices,faces,CData] = Kegel(Teilung)

vertices = zeros(2+Teilung,3);
faces = zeros(2*Teilung,3);

vertices(1+Teilung,:) = [0,0,0];
vertices(2+Teilung,:) = [0,0,1];
for i = 1:Teilung
    phi = 2*pi/Teilung*i;
    vertices(i,:) = [cos(phi),sin(phi),0];
end

for i = 1:(Teilung-1)
    faces(i,:) = [i,i+1,1+Teilung];
    faces(i+Teilung,:) = [i,2+Teilung,i+1];
end
faces(Teilung,:) = [Teilung,1,Teilung+1];
faces(Teilung*2,:) = [Teilung,Teilung+2,1];

CData = 0.5*ones(size(faces));

end

function [vertices, faces, CData] = Zylinder(Teilung)

vertices = zeros(2+2*Teilung,3);
vertices(1,:) = [0,0,0];
vertices(2+Teilung,:) = [0,0,1];

for i = 1:Teilung
    phi = 2*pi/Teilung*i;
    vertices(i+1,:) = [cos(phi),sin(phi),0];
    vertices(i+2+Teilung,:) = [cos(phi),sin(phi),1];
end

faces = zeros(4*Teilung,3);
faces(Teilung,:) = [1,Teilung+1,2];
faces(Teilung*2,:) = [1,2,Teilung+1]+1+Teilung;
faces(Teilung*3,:) = [2,1+Teilung,2+2*Teilung];
faces(Teilung*4,:) = [3+Teilung,2,2+2*Teilung];

for i = 1:Teilung-1
    faces(i,:) = [1,i+1,i+2];
    faces(i+Teilung,:) = [1,i+2,i+1]+1+Teilung;
    faces(i+2*Teilung,:) = [i+2,i+1,i+2+Teilung];
    faces(i+3*Teilung,:) = [i+2+Teilung,i+3+Teilung,i+2];
    
end



CData = 0.5*ones(size(faces));

end

function [V,F,C] = Koordinatensystem(Teilung)

[VKu,FKu,CKu] = Kugel(Teilung);     % Kugel mit Radius 1 erstellen

VKu = VKu*diag([0.9,0.9,0.9]);  % Kugelgröße ändern

[VZ,FZ,CZ] = Zylinder(Teilung);    % Muster-Zylinder erstellen (Radius = 1, Höhe = 1)

CZ1 = 2*CZ*diag([0,0,1]);         % Farbe Zylinder 1 (Z-Achse)
CZ2 = 2*CZ*diag([0,1,0]);         % Farbe Zylinder 2 (Y-Achse)
CZ3 = 2*CZ*diag([1,0,0]);         % Farbe Zylinder 3 (X-Achse)

VZ1 = VZ*diag([0.4,0.4,8]);     % Maße anpassen für Zylinder 1 (Z-Achse)
VZ2 = VZ1*[1,0,0;0,0,1;0,1,0];  % Zylinder 2 erzeugen, durch Drehung von Zylinder 1 (Y-Achse)
VZ3 = VZ1*[0,0,1;0,1,0;1,0,0];  % Zylinder 3 erzeugen, durch Drehung von Zylinder 1 (X-Achse)



[VKe,FKe,CKe] = Kegel(Teilung);     % Muster-Kegel erstellen

CKe1 = 2*CKe*diag([0,0,1]);       % Farbe für Pfeilspitze 1 festlegen (Z-Achse)
CKe2 = 2*CKe*diag([0,1,0]);       % Farbe für Pfeilspitze 2 festlegen (Y-Achse)
CKe3 = 2*CKe*diag([1,0,0]);       % Farbe für Pfeilspitze 3 festlegen (X-Achse)

VKe1 = VKe*diag([0.8,0.8,2])+ones(size(VKe))*diag([0,0,8]); % Pfeilspitze 1 Ausrichten und Maße anpassen
VKe2 = VKe1*[1,0,0;0,0,1;0,1,0];                            % Pfeilspitze 2 Ausrichten durch Drehung von Pfeilspitze 1
VKe3 = VKe1*[0,0,1;0,1,0;1,0,0];                            % Pfeilspitze 2 Ausrichten durch Drehung von Pfeilspitze 1



% Gesamtdaten für Patch erstellen
V = cat(1,VKu,VZ1,VZ2,VZ3,VKe1,VKe2,VKe3);
F = cat(1,FKu,FZ+size(VKu,1),FZ+size(VKu,1)+size(VZ1,1),FZ+size(VKu,1)+2*size(VZ1,1),FKe+size(VKu,1)+3*size(VZ1,1),FKe+size(VKu,1)+3*size(VZ1,1)+size(VKe,1),FKe+size(VKu,1)+3*size(VZ1,1)+2*size(VKe,1));
C = cat(1,CKu,CZ1,CZ2,CZ3,CKe1,CKe2,CKe3);

% Skalieren: Pfeillängen auf 1 setzen
V = V/max(max(V));

end


function [V,F,C] = Pfeil(Teilung)

[VZ,FZ,CZ] = Zylinder(Teilung);    % Muster-Zylinder erstellen (Radius = 1, Höhe = 1)

CZ1 = 2*CZ*diag([0,0,1]);         % Farbe Zylinder 1 (Z-Achse)
VZ1 = VZ*diag([0.4,0.4,7]);     % Maße anpassen für Zylinder 1 (Z-Achse)

[VKe,FKe,CKe] = Kegel(Teilung);     % Muster-Kegel erstellen
CKe1 = 2*CKe*diag([0,0,1]);       % Farbe für Pfeilspitze 1 festlegen (Z-Achse)
VKe1 = VKe*diag([0.8,0.8,3])+ones(size(VKe))*diag([0,0,7]); % Pfeilspitze 1 Ausrichten und Maße anpassen

% Gesamtdaten für Patch erstellen
V = cat(1,VZ1,VKe1);
F = cat(1,FZ,FKe+size(VZ1,1));
C = cat(1,CZ1,CKe1);

% Skalieren: Pfeillängen auf 1 setzen
V = V/max(max(V));

end


function [V,F,C] = Pfeil_mit_Kugel(Teilung)

[VKu,FKu,CKu] = Kugel(Teilung);     % Kugel mit Radius 1 erstellen
VKu = VKu*diag([0.9,0.9,0.9]);  % Kugelgröße ändern
CKu = repmat([1,0,0],size(CKu,1),1);

[VZ,FZ,CZ] = Zylinder(Teilung);    % Muster-Zylinder erstellen (Radius = 1, Höhe = 1)

CZ1 = 2*CZ*diag([1,0,0]);         % Farbe Zylinder 1 (Z-Achse)
VZ1 = VZ*diag([0.4,0.4,8]);     % Maße anpassen für Zylinder 1 (Z-Achse)



[VKe,FKe,CKe] = Kegel(Teilung);     % Muster-Kegel erstellen
CKe1 = 2*CKe*diag([1,0,0]);       % Farbe für Pfeilspitze 1 festlegen (Z-Achse)
VKe1 = VKe*diag([0.8,0.8,2])+ones(size(VKe))*diag([0,0,8]); % Pfeilspitze 1 Ausrichten und Maße anpassen


% Gesamtdaten für Patch erstellen
V = cat(1,VKu,VZ1,VKe1);
F = cat(1,FKu,FZ+size(VKu,1),FKe+size(VKu,1)+size(VZ1,1));
C = cat(1,CKu,CZ1,CKe1);


% Skalieren: Pfeillängen auf 1 setzen
V = V/max(max(V));

end


function [vertices,faces,CData] = Kugel(Teilung)

vertices = zeros(2+Teilung*(ceil(Teilung/2)-1),3);
count = 1;
for j = 1:(ceil(Teilung/2)-1)
    theta = pi/ceil(Teilung/2)*j;
    for i = 1:Teilung
        phi = 2*pi/Teilung*i;
        vertices(count,:) = [sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)];
        count = count + 1;
    end
end
vertices(count,:) = [0,0,1];
vertices(count+1,:) = [0,0,-1];


faces = zeros(Teilung-1,3);

% Deckel oben
for i = 1:Teilung-1
    faces(i,:) = [i,i+1,count];
end
faces = cat(1,faces,[i+1,1,count]);

% Körper
for j = 1:(ceil(Teilung/2)-2)
    for i = 1:Teilung-1
        AddToindex(i,:) = [i+j*Teilung,i+1+(j-1)*Teilung,i+(j-1)*Teilung];
    end
    faces = cat(1,faces,AddToindex);
    faces = cat(1,faces,[(j+1)*Teilung,1+(j-1)*Teilung,i+1+(j-1)*Teilung]);
end
for j = 1:(ceil(Teilung/2)-2)
    for i = 1:Teilung-1
        AddToindex(i,:) = [i+j*Teilung,i+j*Teilung+1,i+1+(j-1)*Teilung];
    end
    faces = cat(1,faces,AddToindex);
    faces = cat(1,faces,[(j+1)*Teilung,1+j*Teilung,1+(j-1)*Teilung]);
end

% Deckel unten
for i = 1:Teilung-1
    faces = cat(1,faces,[(j)*Teilung+i+1,(j)*Teilung+i,count+1]);
end
faces = cat(1,faces,[(j)*Teilung+1,(j+1)*Teilung,count+1]);


CData = 0.5*ones(size(faces));
end

function [vertices,faces,CData] = Wuerfel(Teilung)
    Teilung = 3;
    vertices = zeros((Teilung+1)^3 , 3);
    faces = ones(6 * 2 * Teilung^2,3);
    counter = 0;
    for a = 0:1/Teilung:1
        for b = 0:1/Teilung:1
            for c = 0:1/Teilung:1
                counter = counter +1;
                vertices(counter,:) = [a-0.5, b-0.5, c];
            end
        end
    end
    
    %faces
    t = Teilung;
    for c = 0:t-1
        b = c * (t+1);
        for a = 1:t
            faces((2*t * c) + a,:) =        [b+a      ,b+a+1    ,b+t+a+1  ];
            faces((2*t * c) + t + a,:) =    [b+a+1    ,b+t+a+1  ,b+t+a+2  ];
            temp = (t+1)^2 * t;
            faces(t^2 * 2 + (2*t * c) + a,:) = [temp + b+a      ,temp + b+a+1    ,temp + b+t+a+1  ];
            faces(t^2 * 2 + (2*t * c) + t + a,:) = [temp + b+a+1    ,temp + b+t+a+1  ,temp + b+t+a+2  ];
        end
    end
   
    for b = 1:t
        for a = 1:t
            faces(4 * t^2 + a + (b-1)*2*t, :) = [ 1 + (a-1)*(t+1) + (b-1) * (t+1)^2 , 1 + a * (t+1) + (b-1) * (t+1)^2 , 1 + b*(t+1)^2 + (a-1) * (t+1)];
            faces(4 * t^2 + a + t + (b-1)*2*t, :) = [1 + a*(t+1) + (b-1)*(t+1)^2  , 1 + b*(t+1)^2 + (a-1) * (t+1), 1 + a*(t+1) + b*(t+1)^2];
            temp = t;
            faces(4 * t^2 + a + (b-1)*2*t + 2 * t^2, :) = [temp + 1 + (a-1)*(t+1) + (b-1) * (t+1)^2 , temp + 1 + a * (t+1) + (b-1) * (t+1)^2 , temp + 1 + b*(t+1)^2 + (a-1) * (t+1)];
            faces(4 * t^2 + a + t + (b-1)*2*t + 2 * t^2, :) = [temp + 1 + a*(t+1) + (b-1)*(t+1)^2  , temp + 1 + b*(t+1)^2 + (a-1) * (t+1), temp + 1 + a*(t+1) + b*(t+1)^2];
        end
    end
    
    for b = 1:t
        for a = 1:t
            faces(8 * t^2 + a + (b-1)*2*t, :) = [1 + (a-1) + (b-1)*(t+1)^2, 1 + a + (b-1)*(t+1)^2, 1 + (a-1) + b*(t+1)^2];
            faces(8 * t^2 + a + t + (b-1)*2*t, :) = [1 + a + (b-1)*(t+1)^2, 1 + (a-1) + b*(t+1)^2, 1 + a + b*(t+1)^2];
            temp = (t+1) * t;
            faces(8 * t^2 + a + (b-1)*2*t + 2 * t^2, :) = [temp + 1 + (a-1) + (b-1)*(t+1)^2, temp + 1 + a + (b-1)*(t+1)^2, temp + 1 + (a-1) + b*(t+1)^2];
            faces(8 * t^2 + a + t + (b-1)*2*t + 2 * t^2, :) = [temp + 1 + a + (b-1)*(t+1)^2, temp + 1 + (a-1) + b*(t+1)^2, temp + 1 + a + b*(t+1)^2];
        end
    end
    
    CData = 0.5*ones(size(faces));
end

function [vertices,faces,CData] = Elliptic_cone(Teilung,varargin)
a = 1/2*varargin{1};
b = 1/2*varargin{2};

vertices = zeros(2+Teilung,3);
faces = zeros(2*Teilung,3);

vertices(1+Teilung,:) = [0,0,1];
vertices(2+Teilung,:) = [0,0,0];
for i = 1:Teilung
    phi = 2*pi/Teilung*i;
    vertices(i,:) = [a*cos(phi),b*sin(phi),1];
end

for i = 1:(Teilung-1)
    faces(i,:) = [i,i+1,1+Teilung];
    faces(i+Teilung,:) = [i,2+Teilung,i+1];
end
faces(Teilung,:) = [Teilung,1,Teilung+1];
faces(Teilung*2,:) = [Teilung,Teilung+2,1];

CData = 0.5*ones(size(faces));
end

function [vertices,faces,CData] = Pyramide_updown(~,varargin)
a = 1/2*varargin{1};
b = 1/2*varargin{2};

vertices(1,:) = [a b 1];
vertices(2,:) = [a -b 1];
vertices(3,:) = [-a -b 1];
vertices(4,:) = [-a b 1];
vertices(5,:) = [0 0 0];

i = 1:4;
faces = [i', mod(i',4)+1, 5*ones(4,1);...
    1 2 3; 1 3 4];
CData = 0.5*ones(size(faces));
end