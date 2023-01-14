# class LsiState(object):
#     def __init__(self, hl_state, ml_state, ll_state = None) -> None:
#         self.hl_state = hl_state
#         self.ml_state = ml_state
#         self.ll_state = ll_state
#     """A state in OvercookedGridworld."""
#     def __init__(self, players, objects, order_list):
#         """
#         players: List of PlayerStates (order corresponds to player indices).
#         objects: Dictionary mapping positions (x, y) to ObjectStates. 
#                  NOTE: Does NOT include objects held by players (they are in 
#                  the PlayerState objects).
#         order_list: Current orders to be delivered

#         NOTE: Does not contain time left, which is handled from the environment side.
#         """
#         for pos, obj in objects.items():
#             assert obj.position == pos
#         self.players = tuple(players)
#         self.objects = objects
#         if order_list is not None:
#             assert all(
#                 [o in OvercookedGridworld.ORDER_TYPES for o in order_list])
#         self.order_list = order_list