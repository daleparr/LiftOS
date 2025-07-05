// Authentication store using Zustand

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import Cookies from 'js-cookie';
import { User, Organization, AuthState } from '@/types';
import { authAPI } from '@/lib/api';

interface AuthStore extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  register: (userData: {
    name: string;
    email: string;
    password: string;
    organization_name: string;
  }) => Promise<void>;
  refreshUser: () => Promise<void>;
  setLoading: (loading: boolean) => void;
  clearAuth: () => void;
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      user: null,
      organization: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,

      login: async (email: string, password: string) => {
        try {
          set({ isLoading: true });
          
          const response = await authAPI.login(email, password);
          const { token, user, organization } = response.data.data;
          
          // Store token in cookie
          Cookies.set('auth_token', token, { expires: 7 }); // 7 days
          
          set({
            user,
            organization,
            token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      logout: () => {
        try {
          authAPI.logout().catch(() => {
            // Ignore logout API errors
          });
        } finally {
          Cookies.remove('auth_token');
          set({
            user: null,
            organization: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
          });
        }
      },

      register: async (userData) => {
        try {
          set({ isLoading: true });
          
          const response = await authAPI.register(userData);
          const { token, user, organization } = response.data.data;
          
          // Store token in cookie
          Cookies.set('auth_token', token, { expires: 7 });
          
          set({
            user,
            organization,
            token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      refreshUser: async () => {
        try {
          const token = Cookies.get('auth_token');
          if (!token) {
            get().clearAuth();
            return;
          }

          set({ isLoading: true });
          
          const response = await authAPI.me();
          const { user, organization } = response.data.data;
          
          set({
            user,
            organization,
            token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error) {
          get().clearAuth();
          throw error;
        }
      },

      setLoading: (loading: boolean) => {
        set({ isLoading: loading });
      },

      clearAuth: () => {
        Cookies.remove('auth_token');
        set({
          user: null,
          organization: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
        });
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        organization: state.organization,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

// Helper hooks
export const useAuth = () => {
  const store = useAuthStore();
  return {
    user: store.user,
    organization: store.organization,
    isAuthenticated: store.isAuthenticated,
    isLoading: store.isLoading,
    login: store.login,
    logout: store.logout,
    register: store.register,
    refreshUser: store.refreshUser,
  };
};

export const useUser = () => {
  return useAuthStore((state) => state.user);
};

export const useOrganization = () => {
  return useAuthStore((state) => state.organization);
};

export const useIsAuthenticated = () => {
  return useAuthStore((state) => state.isAuthenticated);
};

// Permission helpers
export const hasPermission = (user: User | null, permission: string, organizationId?: string): boolean => {
  if (!user) return false;
  
  // Admin users have all permissions
  if (user.roles.includes('admin' as any)) return true;
  
  // Check if user has the specific permission
  // This would be expanded based on your RBAC implementation
  return true; // Simplified for now
};

export const hasRole = (user: User | null, role: string): boolean => {
  if (!user) return false;
  return user.roles.includes(role as any);
};

export const canAccessModule = (user: User | null, modulePermissions: string[]): boolean => {
  if (!user) return false;
  if (user.roles.includes('admin' as any)) return true;
  
  // Check if user has any of the required permissions
  return modulePermissions.some(permission => hasPermission(user, permission));
};