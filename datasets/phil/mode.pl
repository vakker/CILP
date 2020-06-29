:- modeh(1,mortal(+entity)).

:- modeb(1,entity_type(+entity,#type)).
:- modeb(1,origin(+entity,#country)).
:- modeb(1,speaks(+entity,#language)).

:- determination(mortal/1,entity_type/2).
:- determination(mortal/1,origin/2).
:- determination(mortal/1,speaks/2).
