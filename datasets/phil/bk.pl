% type definitions
entity(plato). entity(aristotle). entity(socrates).
entity(ares). entity(atlas). entity(zeus).

entity(cicero). entity(seneca). entity(marcus_aurelius).
entity(jupiter). entity(venus). entity(mars).

country(greece). country(rome).
language(greek). language(latin).

type(human). type(god).

% greek humans
entity_type(plato, human).
entity_type(aristotle, human).
entity_type(socrates, human).
% greek gods
entity_type(ares, god).
entity_type(atlas, god).
entity_type(zeus, god).
% roman humans
entity_type(cicero, human).
entity_type(marcus_aurelius, human).
entity_type(seneca, human).
% roman gods
entity_type(venus, god).
entity_type(jupiter, god).
entity_type(mars, god).

% greek humans
origin(socrates, greece).
origin(aristotle, greece).
origin(plato, greece).
% greek gods
origin(zeus, greece).
origin(atlas, greece).
origin(ares, greece).
% roman humans
origin(marcus_aurelius, rome).
origin(seneca, rome).
origin(cicero, rome).
% roman gods
origin(venus, rome).
origin(jupiter, rome).
origin(mars, rome).

speaks(X, greek) :- origin(X, greece).
speaks(X, latin) :- origin(X, rome).
