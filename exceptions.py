class NoFoldAvailableError(Exception):
    """When you cannot find a Fold to allocate a group"""


class NoPostiveInGroupError(Exception):
    """When no positive case is found in a group"""
