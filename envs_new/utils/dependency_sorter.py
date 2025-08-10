"""
Dependency sorting utilities for component-based systems.
Provides generic topological sorting functionality for components with dependencies.
"""
from __future__ import annotations

from typing import Dict, List, Type, Protocol


class HasDependencies(Protocol):
    """Protocol for components that declare dependencies."""
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        """Return list of component names this component depends on."""
        ...


def sort_components_by_dependencies(components: Dict[str, Type[HasDependencies]], 
                                   enabled_components: List[str]) -> List[str]:
    """
    Sort components by dependencies to ensure proper execution order.
    Uses Kahn's topological sorting algorithm.
    
    Args:
        components: Dictionary mapping component names to component classes
        enabled_components: List of component names to sort
        
    Returns:
        List of component names sorted by dependency order
        
    Raises:
        ValueError: If circular dependency is detected
        
    Example:
        >>> components = {
        ...     'frontier': FrontierComponent,
        ...     'agent': AgentComponent,
        ...     'obstacle': ObstacleComponent
        ... }
        >>> enabled = ['frontier', 'agent', 'obstacle']
        >>> sorted_order = sort_components_by_dependencies(components, enabled)
        >>> # Returns: ['frontier', 'agent', 'obstacle'] (based on dependencies)
    """
    # Build dependency graph and calculate in-degrees
    dependency_graph = {}
    in_degree = {}
    
    for name in enabled_components:
        component_class = components[name]
        dependencies = component_class.get_dependencies()
        # Only include dependencies that are actually enabled
        dependencies = [dep for dep in dependencies if dep in enabled_components]
        dependency_graph[name] = dependencies
        in_degree[name] = len(dependencies)
    
    # Kahn's algorithm: start with nodes that have no dependencies
    result = []
    queue = [name for name in enabled_components if in_degree[name] == 0]
    
    while queue:
        current = queue.pop(0)
        result.append(current)
        
        # Remove current node from graph and update in-degrees
        for name, deps in dependency_graph.items():
            if current in deps:
                in_degree[name] -= 1
                if in_degree[name] == 0:
                    queue.append(name)
    
    # Detect circular dependencies
    if len(result) != len(enabled_components):
        remaining = set(enabled_components) - set(result)
        raise ValueError(f"Circular dependency detected among components: {remaining}")
    
    return result


def validate_dependencies(components: Dict[str, Type[HasDependencies]], 
                         enabled_components: List[str]) -> List[str]:
    """
    Validate that all declared dependencies are available.
    
    Args:
        components: Dictionary mapping component names to component classes
        enabled_components: List of enabled component names
        
    Returns:
        List of missing dependencies
        
    Example:
        >>> missing = validate_dependencies(components, ['agent', 'obstacle'])
        >>> if missing:
        ...     print(f"Missing dependencies: {missing}")
    """
    missing_deps = []
    
    for name in enabled_components:
        if name not in components:
            continue
            
        component_class = components[name]
        dependencies = component_class.get_dependencies()
        
        for dep in dependencies:
            if dep not in enabled_components:
                missing_deps.append(f"{name} requires {dep}")
    
    return missing_deps