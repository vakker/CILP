:- modeh(1,drama(+movie)).

:- modeb(*,directed(#director,+movie)).
:- modeb(*,edited(#editor,+movie)).
:- modeb(*,produced(#producer,+movie)).
:- modeb(*,wrote(#writer,+movie)).
:- modeb(*,acted(#actor,+movie)).


:- determination(drama/1,directed/2).
:- determination(drama/1,edited/2).
:- determination(drama/1,produced/2).
:- determination(drama/1,wrote/2).
:- determination(drama/1,acted/2).
:- determination(drama/1,movie/1).
